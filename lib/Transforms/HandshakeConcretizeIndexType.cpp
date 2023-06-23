//===- HandshakeConcretizeIndexType.cpp - Index -> Integer ------*- C++ -*-===//
//
// Implements the --handshake-concretize-index-type pass, which replaces all
// values and attributes of type IndexType with an IntegerType of
// machine-specific width. After changing the types of all SSA values in the IR,
// the pass uses a greedy pattern rewriter to handle operations which require
// special treatment.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

/// Is the type not an index type?
static bool isNotIndexType(Type type) { return !isa<IndexType>(type); };

/// Returns the provided type if it isn't an IndexType, otherwise returns an
/// IntegerType of machine-specific width.
static Type sameOrIndexToInt(Type type) {
  if (isNotIndexType(type))
    return type;
  return IntegerType::get(type.getContext(),
                          IndexType::kInternalStorageBitWidth);
}

namespace {

/// Replaces all IndexType arguments/results in a handshake function's
/// signature.
struct ReplaceFuncSignature : public OpRewritePattern<handshake::FuncOp> {
  using OpRewritePattern<handshake::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    // Check if there is any index type in the function signature
    if (llvm::all_of(funcOp.getArgumentTypes(), isNotIndexType) &&
        llvm::all_of(funcOp.getResultTypes(), isNotIndexType))
      return failure();

    // Recreate a list of function arguments with index types replaced
    SmallVector<Type, 8> argTypes;
    for (auto &argType : funcOp.getArgumentTypes())
      argTypes.push_back(sameOrIndexToInt(argType));

    // Recreate a list of function results with index types replaced
    SmallVector<Type, 8> resTypes;
    for (auto resType : funcOp.getResultTypes())
      resTypes.push_back(sameOrIndexToInt(resType));

    // Replace the function's signature
    rewriter.updateRootInPlace(funcOp, [&] {
      auto funcType = rewriter.getFunctionType(argTypes, resTypes);
      funcOp.setFunctionType(funcType);
    });
    return success();
  }
};

/// Replaces an index cast with an equivalent truncation/extension operations
/// (or with nothing if widths happen to match).
template <typename Op, typename OpExt>
struct ReplaceIndexCast : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op indexCastOp,
                                PatternRewriter &rewriter) const override {
    Value fromVal = indexCastOp.getOperand();
    Value toVal = indexCastOp.getResult();
    unsigned fromWidth = fromVal.getType().getIntOrFloatBitWidth();
    unsigned toWidth = toVal.getType().getIntOrFloatBitWidth();

    if (fromWidth == toWidth)
      // Simply bypass the cast operation if widths match
      rewriter.replaceOp(indexCastOp, fromVal);
    else {
      // Insert an explicit truncation/extension operation to replace the index
      // cast
      rewriter.setInsertionPoint(indexCastOp);
      Operation *castOp;
      if (fromWidth < toWidth)
        castOp = rewriter.create<OpExt>(indexCastOp.getLoc(), toVal.getType(),
                                        fromVal);
      else
        castOp = rewriter.create<arith::TruncIOp>(indexCastOp.getLoc(),
                                                  toVal.getType(), fromVal);
      rewriter.replaceOp(indexCastOp, castOp->getResult(0));
      inheritBB(indexCastOp, castOp);
    }

    return success();
  }
};

/// Replaces the value attribute type in a Handshake constant.
struct ReplaceConstantOpAttr : public OpRewritePattern<handshake::ConstantOp> {
  using OpRewritePattern<handshake::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConstantOp cstOp,
                                PatternRewriter &rewriter) const override {
    if (isa<IndexType>(cstOp.getValueAttr().getType())) {
      cstOp.setValueAttr(IntegerAttr::get(
          IntegerType::get(getContext(), IndexType::kInternalStorageBitWidth),
          cstOp.getValue().cast<IntegerAttr>().getInt()));
      return success();
    }
    return failure();
  }
};

/// Driver for the IndexType concretization pass. First, the type of all SSA
/// values whose type is IndexType is modified to an IntegerType of
/// machine-specific width. Secondly, a greedy pattern rewriter handles
/// operations which require special treatment (e.g. arith.index_cast).
struct HandshakeConcretizeIndexTypePass
    : public HandshakeConcretizeIndexTypeBase<
          HandshakeConcretizeIndexTypePass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    Type indexWidthInt =
        IntegerType::get(ctx, IndexType::kInternalStorageBitWidth);

    // Change the type of all SSA values with an IndexType
    getOperation().walk([&](Operation *op) {
      for (Value operand : op->getOperands())
        if (isa<IndexType>(operand.getType()))
          operand.setType(indexWidthInt);
      for (OpResult result : op->getResults())
        if (isa<IndexType>(result.getType()))
          result.setType(indexWidthInt);
    });

    // Some operations need additional transformation
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns.add<ReplaceFuncSignature,
                 ReplaceIndexCast<arith::IndexCastOp, arith::ExtSIOp>,
                 ReplaceIndexCast<arith::IndexCastUIOp, arith::ExtUIOp>,
                 ReplaceConstantOpAttr>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakeConcretizeIndexType() {
  return std::make_unique<HandshakeConcretizeIndexTypePass>();
}