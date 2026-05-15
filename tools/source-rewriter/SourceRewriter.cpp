//===- SourceRewriter.cpp -------------------------------------------------===//
//
// Standalone clang transformer tool that rewrites the input C file.
//
// Currently only implements one rewrite:
// Turning logical operators (`&&`, `||`)
// into bitwise operators (`&`, `|`)
// with operands wrapped in a `((x) != 0)` comparison.
//
// The C specification requires logical operations to use short-circuiting,
// which means that the rhs of the operation only executes
// if the lhs alone is not enough to know the final value.
//
// We disable this feature, since it blocks HLS developers from control
// of their circuit:
// a && b being logically equivalent to (((a) != 0) & ((b) != 0))
// is not a commonly-known optimization opportunity,
// and so is less likely to be applied manually.
//
// If a HLS developer wishes to have the control-flow dependency
// normally provided by short-circuiting,
// this is easily implemented with an if statement instead.
//
// The effect of this is that logical operations receive both their operands
// as normal data dependencies,
// and there is no control-flow dependency between the two operands.
//
// Docs available at:
//   docs/DeveloperGuide/DynamaticFeaturesAndOptimizations/SourceRewriter.md
//
// Modeled on PointerToRef.cpp from feldsherov/pets-cattle-and-code-examples,
// which is modelled on clang-tools-extra/tool-template.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LogicalResult.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/StandaloneExecution.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"
#include "clang/Tooling/Transformer/Transformer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <memory>
#include <set>
#include <system_error>
#include <utility>
#include <vector>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;
using ::clang::tooling::Transformer;
using ::clang::transformer::applyFirst;
using ::clang::transformer::cat;
using ::clang::transformer::makeRule;
using ::clang::transformer::node;
using ::clang::transformer::RewriteRule;

// The source-file to source-file framework
// we use in this tool uses
// declarative rewrite rules
// that can easily applied to entire codebases
// using boilerplate utility objects/functions
static RewriteRule buildNoShortCircuitRewriteRule() {
  // make a rule to turn a && b into (((a) != 0) & ((b) != 0))
  // binaryOperator matches only operators with two inputs
  // hasOperatorName further filters down to only match &&
  // we then bind the lhs and rhs inputs to the strings
  // 'lhs' and 'rhs' so we can reference them in the output
  //
  // change to then defines the desired output
  auto RuleAnd =
      makeRule(binaryOperator(hasOperatorName("&&"), hasLHS(expr().bind("lhs")),
                              hasRHS(expr().bind("rhs"))),
               changeTo(cat("(((", node("lhs"), ") != 0) & ((", node("rhs"),
                            ") != 0))")));

  // make a rule to turn a || b into (((a) != 0) | ((b) != 0))
  // binaryOperator matches only operators with two inputs
  // hasOperatorName further filters down to only match ||
  // we then bind the lhs and rhs inputs to the strings
  // 'lhs' and 'rhs' so we can reference them in the output
  //
  // change to then defines the desired output
  auto RuleOr =
      makeRule(binaryOperator(hasOperatorName("||"), hasLHS(expr().bind("lhs")),
                              hasRHS(expr().bind("rhs"))),
               changeTo(cat("(((", node("lhs"), ") != 0) | ((", node("rhs"),
                            ") != 0))")));

  // Use a single Transformer to apply both rules
  return applyFirst({RuleAnd, RuleOr});
}

// Once we have ran the Transformer on all input files
// we get an AllChanges, an AtomicChanges object
// storing all the things we should change in the files
//
// Since Transformers cannot apply two rules which overlap
// which occurs often e.g. a && b && c
// we can only allow one change to be applied per file
//
// If we change any file, we then re-run the whole flow
// so that nested expressions are fully handled
//
// If the pass succeeds, it returns a boolean
// indicating whether it made any changes
// Since that means we need to re-run the tool
// due to Transformers not being able to handle nesting
static mlir::FailureOr<bool>
applySourceChanges(const AtomicChanges &AllChanges) {

  // "spec" here means specification
  // and is the options struct for the applyAtomicChanges function
  // We don't enable clean up or formatting here
  // since the change is pretty minor
  // clean up = remove unneeded includes
  tooling::ApplyChangesSpec Spec;
  Spec.Cleanup = false;

  // bool to track where we changed
  // any files
  bool AppliedChanges = false;

  // Set of changes that has only one change
  // per file
  AtomicChanges ChangesToApply;

  // We can only apply one change per file
  // due to Transformers not handling nesting
  std::set<std::string> Files;
  for (const auto &Change : AllChanges) {
    // Add the file to the set of files
    // the .second of the return is true
    // if the file was not already in the set
    if (Files.insert(Change.getFilePath()).second) {
      ChangesToApply.push_back(Change);
    }
  }

  // Iterate over the changes we wish
  // to actually apply
  for (const auto &Change : ChangesToApply) {
    // get the file the change applies to
    auto File = Change.getFilePath();

    // read the file in as a string
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferErr =
        llvm::MemoryBuffer::getFile(File);

    // applySourceChanges fails if we can't open a file
    if (!BufferErr) {
      llvm::errs() << "SourceRewriter Error: failed to open " << File
                   << " for rewriting\n";
      return mlir::failure();
    }

    // Actually apply the change, and return the updated contents
    // as a string
    //
    // We pass the file path only to filter the changes which are applied
    // so the args are [file filter, input string, changes, options struct]
    // We are only passing one change so the filtering is trivial
    auto Result = tooling::applyAtomicChanges(File, (*BufferErr)->getBuffer(),
                                              Change, Spec);

    // applySourceChanges fails if we don't get a string back
    if (!Result) {
      llvm::errs() << toString(Result.takeError());
      return mlir::failure();
    }

    // Build an LLVM file stream object
    // that we can push the new contents into
    std::error_code EC;
    llvm::raw_fd_ostream OS(File, EC, llvm::sys::fs::OF_Text);

    // applySourceChanges fails if building the stream fails
    if (EC) {
      llvm::errs() << EC.message() << "\n";
      return mlir::failure();
    }

    // Push the new file contents into the stream
    // OS doesn't always immediately write to disk
    // but here it will,
    // as it goes out of scope after this line
    // it will write to disk before going out of scope
    OS << *Result;

    // Record that we changed a file
    AppliedChanges = true;
  }

  // If nothing failed,
  // return whether we changed a file or not
  return AppliedChanges;
}

static mlir::FailureOr<bool>
applyRewriteRule(CompilationDatabase &Compilations,
                 const std::vector<std::string> &SourcePaths,
                 const RewriteRule &Rule) {
  // Constructs a standalone clang tool
  // based on the parser options
  //
  // We need to re-construct this after each file edit
  // otherwise there is stale data issues
  //
  // StandaloneToolExecutor contains boilerplate for running
  // the tool you design across an entire code base
  StandaloneToolExecutor Executor(Compilations, SourcePaths);

  // A set of changes to the source file
  // we build the set first, and then apply
  //
  // Each individual change can be quite large,
  // but is applied atomically:
  // Each individual change will succeed or fail,
  // but changes will never be partially applied
  AtomicChanges AllChanges;

  // a LibTooling Transformer is the high-level object
  // used to implement source-to-source rewrites
  //
  // Needs to be in the loop since the callback
  // needs the AllChanges variable
  Transformer TransformerInstance(
      // The rewrite rule itself
      Rule,
      // Callback parameter: what to do once
      // the MatchFinder tells us the rule can be applied
      // and what the rule gave as output
      //
      // We pass AllChanges through the capture list
      // so the lambda sees the local variable
      //
      // The MatchFinder will call the lambda
      // with Changes, a mutable array ref of new changes
      // to apply
      [&AllChanges](
          llvm::Expected<llvm::MutableArrayRef<AtomicChange>> Changes) {
        // if the rule didn't give an output
        // there is some kind of error
        if (!Changes) {
          llvm::errs() << Changes.takeError() << "\n";
          return;
        }

        // otherwise store the rewrite rule output
        // in AllChanges
        AllChanges.insert(AllChanges.end(),
                          std::make_move_iterator(Changes->begin()),
                          std::make_move_iterator(Changes->end()));
      });

  // Create a MatchFinder,
  // and register the Transformer to it
  // so the MatchFinder can use the Transformer's
  // rewrite rule and callback
  // by comparing the rewrite rule to the AST
  //
  // Needs to be in the loop due to the transformer
  // changing
  clang::ast_matchers::MatchFinder MatchFinder;
  TransformerInstance.registerMatchers(&MatchFinder);

  // Run the MatchFinder on all files passed as input
  // which will then populate AllChanges through the callback
  auto Err = Executor.execute(newFrontendActionFactory(&MatchFinder));
  if (Err) {
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
    return mlir::failure();
  }

  // Actually apply AllChanges to the files
  // Returns FailureOr< applied changes >: if any change was applied
  // we need to re-run the whole flow to handle nesting
  return applySourceChanges(AllChanges);
}

// Add a boilerplate message to the help
// of how libtooling CLI tools parse input commands
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// Tool category is used to hide generic libtooling help options
// that are non-specific to this tool
static cl::OptionCategory ToolCategory("source-rewriter options");

int main(int argc, const char **argv) {
  // Handler for if the program crashes
  // to give better errors than normal in c++
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  // parse the CLI args
  // libtooling uses "registration objects",
  // where the declaration of an option object
  // automatically adds it to the parser
  //
  // We still need to pass the ToolCategory
  // to hide the libtooling generic options in
  // the help message
  llvm::Expected<CommonOptionsParser> OptionsParser =
      CommonOptionsParser::create(argc, argv, ToolCategory);

  // if parsing the CLI args failed
  if (!OptionsParser) {
    // print why parsing failed
    llvm::errs() << OptionsParser.takeError();
    return -1;
  }

  // Extract the needed components of the OptionsParser
  // since we need to pass them to multiple executors
  CompilationDatabase &Compilations = OptionsParser->getCompilations();
  const std::vector<std::string> &SourcePaths =
      OptionsParser->getSourcePathList();

  // The declarative rewrite rule is not stateful
  // and so we can declare it outside of the loop
  auto NoShortCircuitRewriteRule = buildNoShortCircuitRewriteRule();

  // If we changed a file,
  // we need to re-run the whole flow
  // to be able to change that file a second time
  // which we need to do to handle nesting
  bool AppliedChanges = true;

  // boolean to track were any changes made
  // to the input kernel
  // If so, we will print a warning
  bool ChangesMade = false;
  while (AppliedChanges) {

    auto Result =
        applyRewriteRule(Compilations, SourcePaths, NoShortCircuitRewriteRule);

    // return that the tool failed
    // if applying the changes failed
    if (mlir::failed(Result))
      return -1;

    // otherwise
    // update the local variable
    // to the boolean returned from applySourceChanges
    AppliedChanges = *Result;

    // Update ChangesMade to be true
    // if AppliedChanges is ever true
    ChangesMade = ChangesMade || AppliedChanges;
  }

  if (ChangesMade) {
    llvm::errs() << "[WARNING] Dynamatic does not use short-circuiting "
                 << "on logical AND (&&) and logical OR (||) operators. "
                 << " Short-circuiting can be enabled by passing "
                 << "--enable-short-circuit to the compile command, "
                 << "but this may come at a performance cost.";
  }
}
