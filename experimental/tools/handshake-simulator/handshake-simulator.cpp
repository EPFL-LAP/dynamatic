#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Support/HandshakeSimulator.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::list<std::string> inputArgs(cl::Positional, cl::desc("<input args>"),
                                       cl::ZeroOrMore, cl::cat(mainCategory));

using namespace dynamatic::experimental;

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "simulator");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
    return 1;
  }

  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect, arith::ArithDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // Assume a single Handshake function in the module
  handshake::FuncOp funcOp = *modOp->getOps<handshake::FuncOp>().begin();

  Simulator sim(funcOp);

  sim.simulate(inputArgs);
  sim.printResults();
}
