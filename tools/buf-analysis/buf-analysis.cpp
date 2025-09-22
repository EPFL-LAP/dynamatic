#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

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
  context.loadDialect<memref::MemRefDialect, handshake::HandshakeDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // We only support one function per module
  handshake::FuncOp funcOp = nullptr;
  for (auto op : modOp->getOps<handshake::FuncOp>()) {
    if (op.isExternal())
      continue;
    if (funcOp) {
      modOp->emitOpError() << "we currently only support one non-external "
                              "handshake function per module";
      return 1;
    }
    funcOp = op;
  }

  int numBufferOps = 0;
  int numSlots = 0;
  int numResources = 0;
  for (auto bufferOp : funcOp.getOps<handshake::BufferOp>()) {
    numBufferOps++;
    numSlots += bufferOp.getNumSlots();
    numResources += bufferOp.getNumSlots() *
                    getHandshakeTypeBitWidth(bufferOp.getResult().getType());
  }

  std::cout << "Number of Buffer Ops: " << numBufferOps << "\n";
  std::cout << "Number of Slots: " << numSlots << "\n";
  std::cout << "Number of Resources: " << numResources << "\n";

  return 0;
}