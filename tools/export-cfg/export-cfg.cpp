#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

#include <optional>
#include <regex>
#include <string>

using namespace llvm;
using namespace mlir;

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

std::string getCfgNodeLabel(Block &block) {
  std::string blockLabel;
  llvm::raw_string_ostream bl(blockLabel);
  bl << "\"";
  block.printAsOperand(bl);
  bl << "\"";
  bl.flush();
  return bl.str();
}

std::optional<int> extractFirstNumber(const std::string &input) {
  static const std::regex numberRegex(R"(\d+)");
  std::smatch match;
  if (std::regex_search(input, match, numberRegex)) {
    return std::stoi(match.str());
  }
  return std::nullopt;
}

std::string getEdgeColor(Block &pred, Block &succ) {
  std::optional<int> predNumber = extractFirstNumber(getCfgNodeLabel(pred));
  std::optional<int> succNumber = extractFirstNumber(getCfgNodeLabel(succ));
  if (predNumber && succNumber) {
    if (*predNumber >= *succNumber) {
      return "red";
    }
  }
  return "black";
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Exports a DOT graph corresponding to the module for visualization\n"
      "and legacy-compatibility purposes.The pass only supports exporting\n"
      "the graph of a single Handshake function at the moment, and will fail\n"
      "if there is more than one Handhsake function in the module.");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFileName
                 << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er) level
  // dialects or parsers. Allow unregistered dialects to not fail in these
  // cases
  MLIRContext context;
  context.loadDialect<memref::MemRefDialect, arith::ArithDialect,
                      cf::ControlFlowDialect, math::MathDialect,
                      func::FuncDialect, LLVM::LLVMDialect>();
  context.allowUnregisteredDialects();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // We only support one function per module
  func::FuncOp funcOp = nullptr;
  for (auto op : modOp->getOps<func::FuncOp>()) {
    if (op.isExternal())
      continue;
    if (funcOp) {
      modOp->emitOpError() << "we currently only support one non-external "
                              "handshake function per module";
      return 1;
    }
    funcOp = op;
  }

  mlir::raw_indented_ostream os(llvm::outs());

  os << "digraph \"" << funcOp.getName() << "\" {\n";
  os.indent();
  os << "node [shape=box];\n";

  for (Block &block : funcOp.getBlocks()) {
    std::string blockLabel;
    llvm::raw_string_ostream bl(blockLabel);

    os << getCfgNodeLabel(block) << " [label=" << getCfgNodeLabel(block)
       << ", shape=\"oval\"];\n";

    Operation *terminator = block.getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
      os << getCfgNodeLabel(block) << " -> " << getCfgNodeLabel(*br.getDest())
         << "[color=\"" << getEdgeColor(block, *br.getDest()) << "\"];\n";
    } else if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
      os << getCfgNodeLabel(block) << " -> "
         << getCfgNodeLabel(*condBr.getTrueDest())
         << "[label=\"true\", color=\""
         << getEdgeColor(block, *condBr.getTrueDest()) << "\"];\n";
      os << getCfgNodeLabel(block) << " -> "
         << getCfgNodeLabel(*condBr.getFalseDest())
         << "[label=\"false\", color=\""
         << getEdgeColor(block, *condBr.getFalseDest()) << "\"];\n";
    }
  }
  os.unindent();

  os << "}\n";
  os.flush();

  return 0;
}
