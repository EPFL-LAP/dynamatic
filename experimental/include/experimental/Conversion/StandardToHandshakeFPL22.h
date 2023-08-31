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
#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
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
  CFGLoop *loop = nullptr;
  bool isHeader = false;
  bool isExit = false;
  bool isLatch = false;
};

/// Function that runs loop analysis on the funcOp Region.
DenseMap<Block *, BlockLoopInfo> findLoopDetails(CFGLoopInfo &li,
                                                 Region &funcReg);

/// Find CFGLoop which is the least common ancestor loop. If not found, nullptr
/// is returned.
CFGLoop *findLCALoop(CFGLoop *innermostLoopOfBB1, CFGLoop *innermostLoopOfBB2);

// This class is used to inherit from FPGA18's standard-to-handshake lowering
// infrastructure and implementation while providing us a way to
// change/add/remove/reorder specific conversion steps to match FPL22's elastic
// pass.
class HandshakeLoweringFPL22 : public HandshakeLoweringFPGA18 {
public:
  /// Constructor simply forwards its arguments to the parent class.
  explicit HandshakeLoweringFPL22(Region &r) : HandshakeLoweringFPGA18(r) {}

  /// Adding a control-only argument to the region's entry block and connecting
  /// the start control to all blocks.
  LogicalResult createStartCtrl(ConversionPatternRewriter &rewriter);

  /// Preventing a token missmatch by adding additional merges to the loop
  /// header block.
  LogicalResult handleTokenMissmatch(
      DenseMap<Value, std::set<Block *>> &valueIsConsumedInBlocksMap,
      std::set<Operation *> &preventTokenMissmatchMerges,
      BackedgeBuilder &edgeBuilder, ConversionPatternRewriter &rewriter);

  /// Inserts a merge for a block argument.
  MergeOpInfo insertMerge(Block *block, Value val, BackedgeBuilder &edgeBuilder,
                          ConversionPatternRewriter &rewriter);

  /// Inserts SSA merges for all block arguments.
  BlockOps insertMergeOps(ValueMap &mergePairs, BackedgeBuilder &edgeBuilder,
                          ConversionPatternRewriter &rewriter);

  /// Adding both SSA merges and merges used to prevent token missmatch.
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