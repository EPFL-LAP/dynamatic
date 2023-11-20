//===- HandshakeConcretizeIndexType.cpp - Index -> Integer ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-concretize-index-type pass, which replaces all
// values and attributes of type IndexType with an IntegerType of
// machine-specific width. After changing the types of all SSA values in the IR,
// the pass uses a greedy pattern rewriter to handle operations which require
// special treatment.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "dynamatic/Analysis/ConstantAnalysis.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/HandshakeMinimizeCstWidth.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

/// Is the type not an index type?
static bool isNotIndexType(Type type) { return !isa<IndexType>(type); };

/// Returns the provided type if it isn't an IndexType, otherwise returns an
/// IntegerType of the provided width.
static Type sameOrIndexToInt(Type type, unsigned width) {
  if (isNotIndexType(type))
    return type;
  return IntegerType::get(type.getContext(), width);
}

LogicalResult dynamatic::verifyAllIndexConcretized(Operation *op) {
  if (!llvm::all_of(op->getOperandTypes(), isNotIndexType) ||
      !llvm::all_of(op->getResultTypes(), isNotIndexType))
    return op->emitError()
           << "Operation has at least one index-typed operand or result.";
  return success();
}

LogicalResult dynamatic::verifyAllIndexConcretized(handshake::FuncOp funcOp) {
  // Check the function signature
  if (!llvm::all_of(funcOp.getArgumentTypes(), isNotIndexType) ||
      !llvm::all_of(funcOp.getResultTypes(), isNotIndexType))
    return funcOp.emitError()
           << "Function has at least one index-typed argument or result.";

  // Check all operations inside the function
  // NOTE: (lucas) Using a for loop instead of llvm::all_of here as the fact
  // that verifyAllIndexConcretized is overloaded trips out type inference and
  // fails to compile the latter
  // NOLINTNEXTLINE(readability-use-anyofallof)
  for (Operation &op : funcOp.getOps())
    if (failed(verifyAllIndexConcretized(&op)))
      return failure();
  return success();
}

LogicalResult dynamatic::verifyAllIndexConcretized(ModuleOp modOp) {
  // Check all functions inside the module
  // NOTE: (lucas) Using a for loop instead of llvm::all_of here as the fact
  // that verifyAllIndexConcretized is overloaded trips out type inference and
  // fails to compile the latter
  // NOLINTNEXTLINE(readability-use-anyofallof)
  for (auto funcOp : modOp.getOps<handshake::FuncOp>())
    if (failed(verifyAllIndexConcretized(funcOp)))
      return failure();
  return success();
}

namespace {

/// Replaces all IndexType arguments/results in a handshake function's
/// signature.
struct ReplaceFuncSignature : public OpRewritePattern<handshake::FuncOp> {
  using OpRewritePattern<handshake::FuncOp>::OpRewritePattern;

  ReplaceFuncSignature(MLIRContext *ctx, unsigned width)
      : OpRewritePattern<handshake::FuncOp>(ctx), width(width) {}

  LogicalResult matchAndRewrite(handshake::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    // Check if there is any index type in the function signature
    if (llvm::all_of(funcOp.getArgumentTypes(), isNotIndexType) &&
        llvm::all_of(funcOp.getResultTypes(), isNotIndexType))
      return failure();

    // Recreate a list of function arguments with index types replaced
    SmallVector<Type, 8> argTypes;
    for (auto &argType : funcOp.getArgumentTypes())
      argTypes.push_back(sameOrIndexToInt(argType, width));

    // Recreate a list of function results with index types replaced
    SmallVector<Type, 8> resTypes;
    for (auto resType : funcOp.getResultTypes())
      resTypes.push_back(sameOrIndexToInt(resType, width));

    // Replace the function's signature
    rewriter.updateRootInPlace(funcOp, [&] {
      auto funcType = rewriter.getFunctionType(argTypes, resTypes);
      funcOp.setFunctionType(funcType);
    });
    return success();
  }

private:
  /// The width to concretize IndexType's with.
  unsigned width;
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
      // Insert an explicit truncation/extension operation to replace the
      // index cast
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

  ReplaceConstantOpAttr(MLIRContext *ctx, unsigned width)
      : OpRewritePattern<handshake::ConstantOp>(ctx), width(width) {}

  LogicalResult matchAndRewrite(handshake::ConstantOp cstOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<IndexType>(cstOp.getValueAttr().getType()))
      return failure();

    auto newAttr =
        IntegerAttr::get(IntegerType::get(getContext(), width),
                         cstOp.getValue().cast<IntegerAttr>().getInt());
    cstOp.setValueAttr(newAttr);

    // Check whether index concretization created a duplicated constant; if
    // so, delete the duplicate
    if (auto otherCstOp = findEquivalentCst(cstOp))
      rewriter.replaceOp(cstOp, otherCstOp.getResult());

    return success();
  }

private:
  /// The width to concretize IndexType's with.
  unsigned width;
};

/// Driver for the IndexType concretization pass. First, the type of all SSA
/// values whose type is IndexType is modified to an IntegerType of
/// machine-specific width. Secondly, a greedy pattern rewriter handles
/// operations which require special treatment (e.g. arith.index_cast).
struct HandshakeConcretizeIndexTypePass
    : public dynamatic::impl::HandshakeConcretizeIndexTypeBase<
          HandshakeConcretizeIndexTypePass> {

  HandshakeConcretizeIndexTypePass(unsigned width) { this->width = width; }

  void runDynamaticPass() override {
    auto *ctx = &getContext();
    Type indexWidthInt = IntegerType::get(ctx, width);

    // Change the type of all SSA values with an IndexType
    WalkResult walkRes = getOperation().walk([&](Operation *op) {
      for (Value operand : op->getOperands())
        if (isa<IndexType>(operand.getType()))
          operand.setType(indexWidthInt);
      for (OpResult result : op->getResults())
        if (isa<IndexType>(result.getType()))
          result.setType(indexWidthInt);

      if (auto cstOp = dyn_cast<handshake::ConstantOp>(op)) {
        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
          // Constants must still be able to fit their value in the new width
          unsigned requiredWidth = computeRequiredBitwidth(intAttr.getValue());
          if (requiredWidth > width) {
            cstOp.emitError()
                << "constant value " << intAttr.getValue().getZExtValue()
                << " does not fit on " << width << " bits (requires at least "
                << requiredWidth << ")";
            return WalkResult::interrupt();
          }
        }
      }

      return WalkResult::advance();
    });

    if (walkRes.wasInterrupted())
      return signalPassFailure();

    // Some operations need additional transformation
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns.add<ReplaceFuncSignature, ReplaceConstantOpAttr>(ctx, width);
    patterns.add<ReplaceIndexCast<arith::IndexCastOp, arith::ExtSIOp>,
                 ReplaceIndexCast<arith::IndexCastUIOp, arith::ExtUIOp>>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeConcretizeIndexType(unsigned width) {
  return std::make_unique<HandshakeConcretizeIndexTypePass>(width);
}