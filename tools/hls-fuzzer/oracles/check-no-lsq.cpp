
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/Tools/ParseUtilities.h"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"

using namespace mlir;
using namespace dynamatic;

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "expected exactly one argument\n";
    return -1;
  }

  StringRef mlirFile = argv[1];
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFileOrSTDIN(mlirFile, true);
  if (!buffer) {
    llvm::errs() << "failed to open" << mlirFile << "\n";
    return -1;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(*buffer), SMLoc());
  DialectRegistry registry;
  registry.insert<handshake::HandshakeDialect, arith::ArithDialect>();
  MLIRContext context(registry);
  ParserConfig config(&context);
  OwningOpRef<Operation *> module =
      parseSourceFileForTool(sourceMgr, config, true);

  WalkResult result = module->walk([&](handshake::LSQOp) {
    llvm::errs() << "IR must not contain an LSQ\n";
    return WalkResult::interrupt();
  });
  if (result.wasInterrupted())
    return -1;

  return 0;
}
