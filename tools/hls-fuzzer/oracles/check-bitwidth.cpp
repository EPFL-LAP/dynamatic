
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/Tools/ParseUtilities.h"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <dynamatic/Dialect/Handshake/HandshakeOps.h>
#include <dynamatic/Dialect/Handshake/HandshakeTypes.h>

using namespace mlir;
using namespace dynamatic;

int main(int argc, char **argv) {
  if (argc != 3) {
    llvm::errs() << "expected exactly two arguments\n";
    return -1;
  }

  StringRef mlirFile = argv[1];
  StringRef bitwidthArg = argv[2];
  std::uint32_t bitWidth;
  if (bitwidthArg.getAsInteger(0, bitWidth)) {
    llvm::errs() << "expected an integer as second argument\n";
    return -1;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFileOrSTDIN(mlirFile, true);
  if (!buffer) {
    llvm::errs() << "failed to open" << mlirFile << "\n";
    return -1;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(*buffer), SMLoc());
  DialectRegistry registry;
  registry.insert<handshake::HandshakeDialect>();
  MLIRContext context(registry);
  ParserConfig config(&context);
  OwningOpRef<Operation *> module =
      parseSourceFileForTool(sourceMgr, config, true);

  WalkResult result = module->walk([&](Operation *op) {
    for (Value iter : op->getResults()) {
      auto channelType = dyn_cast<handshake::ChannelType>(iter.getType());
      if (!channelType || !isa<IntegerType>(channelType.getDataType()))
        continue;

      // Results that feed into 'end' ops are allowed to use a higher bitwidth
      // as it is required for interface conformance.
      if (llvm::any_of(iter.getUsers(),
                       [](Operation *op) { return isa<handshake::EndOp>(op); }))
        continue;

      if (channelType.getDataBitWidth() > bitWidth) {
        op->emitError("expected computation with at most a bitwidth of '")
            << bitWidth << "' rather than '" << channelType.getDataBitWidth()
            << "'";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return -1;

  return 0;
}
