#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang;

namespace {

// duplicate mlir logical result for clarity without
// needing the dependency
struct LogicalResult {
  bool IsFailure;
};
static LogicalResult success() { return {false}; }
static LogicalResult failure() { return {true}; }
static bool failed(LogicalResult R) { return R.IsFailure; }

// emit an error with the given message at the given token
// through the prepocessor diagnostic system
//
// normally diagnostic IDs are hardcoded so that error messages are shared
// between multiple call sites, but here we register a new one for each
// error message on the fly
static void error(Preprocessor &PP, const Token &Tok, const char *errorMsg) {
  unsigned ID = PP.getDiagnostics().getDiagnosticIDs()->getCustomDiagID(
      DiagnosticIDs::Error, errorMsg);
  PP.Diag(Tok, ID);
}

// struct to store the information
// we get from a speculate pragma
struct SpeculatePragmaInfo {
  std::string Variable;
  uint64_t MaxPredictions = 0;
  std::string Style;
  SourceLocation PragmaLoc;
};

// PragmaHandler to handle the speculate pragma
class SpeculatePragmaHandler : public PragmaHandler {
public:
  // matches to #pragma DYN speculate
  SpeculatePragmaHandler() : PragmaHandler("speculate") {}

  // what to do when the pragma is encountered
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &FirstToken) override {
    // initialize the struct we use to store
    // the pragma information
    SpeculatePragmaInfo SpecPragmaInfo;

    // populate the fields of SpecPragmaInfo
    // by parsing the pragma text
    if (failed(parseOptions(PP, FirstToken, SpecPragmaInfo)))
      return;

    // then output the injected
    // speculate function decl and call
    emitInjectedTokens(PP, SpecPragmaInfo);
  }

private:
  // Using the preprocessor to lex in the pragma
  // and convert it to the ParsedOptions struct
  LogicalResult parseOptions(Preprocessor &PP, const Token &FirstToken,
                             SpeculatePragmaInfo &SpecPragmaInfo) {

    // store the location of the pragma
    SpecPragmaInfo.PragmaLoc = FirstToken.getLocation();

    // booleans to track which info we have received
    bool sawVariable = false, sawMaxPred = false, sawStyle = false;

    // store output of lexer
    Token Tok;

    // lex the initial token for the first iteration
    PP.Lex(Tok);

    // while the lexer has input from this line
    while (Tok.isNot(tok::eod)) {
      // we expect a named option
      if (Tok.isNot(tok::identifier)) {
        error(PP, Tok, "expected named option in #pragma DYN speculate");
        return failure();
      }

      // get the value of the named option
      llvm::StringRef Name = Tok.getIdentifierInfo()->getName();

      // lex the = sign
      PP.Lex(Tok);
      if (Tok.isNot(tok::equal)) {
        error(PP, Tok, "expected '=' after option name");
        return failure();
      }

      if (Name == "variable") {
        // if the named option is variable
        // lex the variable name
        PP.Lex(Tok);

        // enforce that variable name
        // lexes properly
        if (Tok.isNot(tok::identifier)) {
          error(PP, Tok, "expected variable name after variable=");
          return failure();
        }

        // store the variable name info
        SpecPragmaInfo.Variable = Tok.getIdentifierInfo()->getName().str();
        sawVariable = true;

        // lex the initial token for the next iteration
        PP.Lex(Tok);

      } else if (Name == "max_predictions") {
        // if the named option is max predictions
        // lex the number of max predictions
        PP.Lex(Tok);

        // try to store the value lexed from the pragma
        // into SpecPragmaInfo.MaxPredictions
        if (Tok.isNot(tok::numeric_constant) ||
            !PP.parseSimpleIntegerLiteral(Tok, SpecPragmaInfo.MaxPredictions)) {
          error(PP, Tok, "expected integer literal after max_predictions=");
          return failure();
        }
        sawMaxPred = true;

        // no lex since parseSimpleIntegerLiteral updated Tok
        // already
      } else if (Name == "style") {
        // if the named option is style
        // lex the style identifier
        PP.Lex(Tok);

        // enforce that style name
        // lexes properly
        if (Tok.isNot(tok::identifier)) {
          error(PP, Tok, "expected identifier after style=");
          return failure();
        }

        // store the variable name info
        SpecPragmaInfo.Style = Tok.getIdentifierInfo()->getName().str();

        // currently style has to be standard,
        // will whitelist more options later
        if (SpecPragmaInfo.Style != "standard") {
          error(PP, Tok, "style must be standard");
          return failure();
        }
        sawStyle = true;

        // lex the initial token for the next iteration
        PP.Lex(Tok);
      } else {
        // otherwise we have gotten an unknown named option
        error(PP, Tok, "unknown option in #pragma DYN speculate");
        return failure();
      }
    }

    // Enforce that we got every option needed
    if (!sawVariable || !sawMaxPred || !sawStyle) {
      error(PP, FirstToken,
            "#pragma DYN speculate requires variable=, "
            "max_predictions=, style=");
      return failure();
    }
    return success();
  }

  void emitInjectedTokens(Preprocessor &PP,
                          const SpeculatePragmaInfo &SpecPragmaInfo) {
    // Inspired by how the preprocessor injects
    // the content of header files
    // we create a fake header file with a function dec
    // and function call
    //
    // and replace the pragma with it
    //
    // We will later convert this function call
    // to an mlir attribute, before running
    // CFToHandshake

    // Inject declaration of a custom speculate function
    // Take a double as the input variable, as any numeric value can be cast to
    // double and produce a int as output, as any numeric value can be cast to
    // from int
    std::string Decl = "extern int __dyn_speculate(double, int, const char *) "
                       "__attribute__((noinline, noduplicate));\n";

    // produce a call of the spec function on the value being speculate on
    // so that x = spec(x, [other options for speculator])
    // this guarantees that x will exist in the final circuit
    // which is required for the
    // manually specified prediction method of x
    // to be usable
    std::string Call = llvm::formatv(
        "{0} = __dyn_speculate({0}, {1}, \"{2}\");\n", SpecPragmaInfo.Variable,
        SpecPragmaInfo.MaxPredictions, SpecPragmaInfo.Style);

    // LLVM reads files into memory buffers
    // before processing them
    // so we place our fake file into one here
    // and give it the name "<dyn-speculate-injected>"
    // which is purely for error messages
    auto MB = llvm::MemoryBuffer::getMemBufferCopy(Decl + Call,
                                                   "<dyn-speculate-injected>");

    // Here we prep the fake file to be a
    // fake header file
    // (owned by the user and treated as unloaded)
    // and say the pragma triggered its include
    FileID FID = PP.getSourceManager().createFileID(
        std::move(MB), SrcMgr::C_User, /*LoadedID=*/0,
        /*LoadedOffset=*/0, SpecPragmaInfo.PragmaLoc);

    // And then we tell the preprocessor
    // to process the fake header file next
    // in the same way #include directives work
    PP.EnterSourceFile(FID, /*DirLookup=*/nullptr, SpecPragmaInfo.PragmaLoc);
  }
};

// Declare the DynPragmaNamespace object
// which stores all of our Dyn pragmas
class DynPragmaNamespace : public PragmaNamespace {
public:
  // set the "name" of the namespace to DYN
  DynPragmaNamespace() : PragmaNamespace("DYN") {
    // put the speculate pragma in the DYN namespace
    AddPragma(new SpeculatePragmaHandler());
  }
};

} // namespace

// Registration object which adds the DynPragmaNamespace
// to clang's PragmaHandlerRegistry
// so that clang nows how to handle pragmas in
// the DYN pragma namespace
static PragmaHandlerRegistry::Add<DynPragmaNamespace>
    X("DYN", "Dynamatic pragma namespace");
