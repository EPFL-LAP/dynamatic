//===- StandardToHandshakeFPGA18.h - FPGA18's elastic pass ------*- C++ -*-===//
//
// This file declares the --lower-std-to-handshake-fpga18 conversion pass along
// with a helper class for performing the lowering.
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

/// Operation attribute to identify the basic block the operation originated
/// from in the std-level IR.
const std::string BB_ATTR = "bb";

// This class is used to inherit from CIRCT's standard-to-handshake lowering
// infrastructure and implementation while providing us a way to
// change/add/remove/reorder specific conversion steps to match FPGA18's elastic
// pass.
class HandshakeLoweringFPGA18 : public HandshakeLowering {
public:
  /// Used to store a list of operations grouped by their parent basic block.
  /// Defined with a SmallVector instead of a DenseMap to ensure deterministic
  /// iteration order.
  using MemBlockOps = SmallVector<std::pair<Block *, std::vector<Operation *>>>;

  /// Used to store a "mapping" between memrefs and the set of operations
  /// referencing them, grouped by their parent block. Defined with a
  /// SmallVector instead of a DenseMap to ensure deterministic iteration order.
  using MemInterfacesInfo = SmallVector<std::pair<Value, MemBlockOps>>;

  /// Constructor simply forwards its arguments to the parent class.
  explicit HandshakeLoweringFPGA18(Region &r) : HandshakeLowering(r) {}

  /// Creates the control-only network by adding a control-only argument to the
  /// region's entry block and forwarding it through all basic blocks.
  LogicalResult createControlOnlyNetwork(ConversionPatternRewriter &rewriter);

  /// Identifies all memory interfaces and operations in the function, replaces
  /// all load/store-like operations by their handshake counterparts, and fills
  /// memInfo with information about which operations use which interface.
  LogicalResult replaceMemoryOps(ConversionPatternRewriter &rewriter,
                                 MemInterfacesInfo &memInfo);

  /// Instantiates all memory interfaces and connects them to their respective
  /// load/store operations.
  LogicalResult connectToMemory(ConversionPatternRewriter &rewriter,
                                MemInterfacesInfo &memInfo);

  /// Sets an integer "bb" attribute on each operation to identify the basic
  /// block from which the operation originates in the std-level IR.
  LogicalResult idBasicBlocks(ConversionPatternRewriter &rewriter);

  /// Creates the region's return network by sequentially moving all blocks'
  /// operations to the entry block, replacing func::ReturnOp's with
  /// handshake::ReturnOp's, deleting all block terminators and non-entry
  /// blocks, merging the results of all return statements, and creating the
  /// region's end operation.
  LogicalResult createReturnNetwork(ConversionPatternRewriter &rewriter,
                                    bool idBasicBlocks);
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createStandardToHandshakeFPGA18Pass(bool idBasicBlocks = false);

/// This struct groups the operations of a handshake::FuncOp in "blocks" based
/// on the "bb" attribute potentially attached to each operation.
struct HandshakeBlocks {
  /// Maps each block ID to the operations (in program order) that ate tagged
  /// with it.
  DenseMap<unsigned, SmallVector<Operation *>> blocks;
  /// List of operations (in program order) that do not belong to any block
  SmallVector<Operation *> outOfBlocks;
};

/// Groups the operations of a function into "blocks" based on the "bb"
/// attribute of each operation.
HandshakeBlocks getHandshakeBlocks(handshake::FuncOp funcOp);

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_STANDARDTOHANDSHAKEFPGA18_H
