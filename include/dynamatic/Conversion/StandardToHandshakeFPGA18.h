//===- StandardToHandshakeFPGA18.h ----------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_STANDARDTOHANDSHAKEFPGA18_H
#define DYNAMATIC_CONVERSION_STANDARDTOHANDSHAKEFPGA18_H

#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

using namespace circt;
using namespace circt::handshake;

namespace dynamatic {
class HandshakeLoweringFPGA18 : public HandshakeLowering {
public:
  // Using SmallVectors instead of DenseMaps to ensure deterministic iteration
  // order in connectToMemory
  using MemBlockOps = SmallVector<std::pair<Block *, std::vector<Operation *>>>;
  using MemInterfacesInfo = SmallVector<std::pair<Value, MemBlockOps>>;

  explicit HandshakeLoweringFPGA18(Region &r) : HandshakeLowering(r) {}

  LogicalResult createControlOnlyNetwork(ConversionPatternRewriter &rewriter);

  LogicalResult replaceMemoryOps(ConversionPatternRewriter &rewriter,
                                 MemInterfacesInfo &memInfo);

  LogicalResult connectToMemory(ConversionPatternRewriter &rewriter,
                                MemInterfacesInfo &memInfo);

  LogicalResult createReturnNetwork(ConversionPatternRewriter &rewriter);
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createStandardToHandshakeFPGA18Pass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_STANDARDTOHANDSHAKEFPGA18_H
