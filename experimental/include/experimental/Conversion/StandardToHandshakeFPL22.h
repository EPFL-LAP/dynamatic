//===- StandardToHandshakeFPL22.h - FPL22's elastic pass ------*- C++ -*-===//
//
// This file declares the --exp-lower-std-to-handshake-fpl22 conversion pass
// along with a helper class for performing the lowering.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_CONVERSION_STANDARDTOHANDSHAKEFPL22_H
#define EXPERIMENTAL_CONVERSION_STANDARDTOHANDSHAKEFPL22_H

#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;

namespace dynamatic {
namespace experimental {

/// Structure that stores loop information of a Block.
struct BlockLoopInfo {
  CFGLoop *loop;
  bool isHeader = false;
  bool isExit = false;
  bool isLatch = false;
};

/// Function that runs a loop analysis on the funcOp Region.
DenseMap<Block *, BlockLoopInfo> findLoopDetails(func::FuncOp &funcOp);

// This class is used to inherit from CIRCT's standard-to-handshake lowering
// infrastructure and implementation while providing us a way to
// change/add/remove/reorder specific conversion steps to match FPL22's elastic
// pass.
class HandshakeLoweringFPL22 : public HandshakeLowering {
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
  explicit HandshakeLoweringFPL22(Region &r) : HandshakeLowering(r) {}

  /// Adding a control-only argument to the region's entry block and connecting
  /// the start control to all blocks.
  LogicalResult createStartCtrl(ConversionPatternRewriter &rewriter);

  /// Identifies all memory interfaces and operations in the function, replaces
  /// all load/store-like operations by their handshake counterparts, and fills
  /// memInfo with information about which operations use which interface.
  LogicalResult replaceMemoryOps(ConversionPatternRewriter &rewriter,
                                 MemInterfacesInfo &memInfo);

  /// Instantiates all memory interfaces and connects them to their respective
  /// load/store operations.
  LogicalResult connectToMemory(ConversionPatternRewriter &rewriter,
                                MemInterfacesInfo &memInfo);

  /// Replaces undefined operations (mlir::LLVM::UndefOp) with a default "0"
  /// constant triggered by the enclosing block's control merge.
  LogicalResult replaceUndefinedValues(ConversionPatternRewriter &rewriter);

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

  // TODO: add missing descriptions, and discuss code organization with Lucas

  LogicalResult handleTokenMissmatch(ConversionPatternRewriter &rewriter);

  MergeOpInfo insertMerge(Block *block, Value val, BackedgeBuilder &edgeBuilder,
                          ConversionPatternRewriter &rewriter);

  BlockOps insertMergeOps(ValueMap &mergePairs, BackedgeBuilder &edgeBuilder,
                          ConversionPatternRewriter &rewriter);

  LogicalResult addMergeOps(ConversionPatternRewriter &rewriter);
};

#define GEN_PASS_DECL_STANDARDTOHANDSHAKEFPL22
#define GEN_PASS_DEF_STANDARDTOHANDSHAKEFPL22
#include "experimental/Conversion/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createStandardToHandshakeFPL22Pass(bool idBasicBlocks = false);

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_CONVERSION_STANDARDTOHANDSHAKEFPL22_H