//===- ArithReduceStrength.cpp - Reduce stregnth of arith ops ---*- C++ -*-===//
//
// Implements the --arith-reduce-strength pass, which greedily applies rewrite
// patterns to arithmetic operations to reduce their strength, improving
// performance and/or area.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/ArithReduceStrength.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace dynamatic;
using namespace circt::handshake;

/// Determines whether the defining operation of a value is a constant -1.
static bool isConstantNegOne(Value val) {
  if (auto cstOp = val.getDefiningOp<arith::ConstantOp>())
    if (auto cstAttr = dyn_cast<IntegerAttr>(cstOp.getValue()))
      return cstAttr.getValue().getSExtValue() == -1;
  return false;
}

/// Determines whether the defining operation of a value is a multiplication
/// with a constant -1.
static Value isMulTimesNegOne(Value val) {
  if (auto mulOp = val.getDefiningOp<arith::MulIOp>()) {
    // Check whether one of the two operands is a constant -1 value. If yes,
    // return the other operand
    auto mulOperands = mulOp->getOperands();
    auto mul0 = mulOperands[0], mul1 = mulOperands[1];
    if (isConstantNegOne(mul0))
      return mul1;
    if (isConstantNegOne(mul1))
      return mul0;
  }
  return nullptr;
}

namespace {

/// Replaces addition of the form `x + (-y)` into an equivalent `x - y`
/// substraction. This can end up indirectly removing a multiplication from the
/// IR via DCE when the result of the multiplication computing `-y` isn't used
/// anywhere else.
struct ReplaceMulAddWithSub : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rewriter) const override {
    auto addOperands = addOp->getOperands();
    auto add0 = addOperands[0], add1 = addOperands[1];

    // Check whether any operand of the addition is the result of a
    // multiplication with a constant -1. If yes, replace the addition with an
    // equivalent substraction
    if (auto newOperand = isMulTimesNegOne(add0)) {
      rewriter.replaceOp(
          addOp,
          rewriter.create<arith::SubIOp>(addOp->getLoc(), add1, newOperand)
              ->getResults());
      return success();
    }
    if (auto newOperand = isMulTimesNegOne(add1)) {
      rewriter.replaceOp(
          addOp,
          rewriter.create<arith::SubIOp>(addOp->getLoc(), add0, newOperand)
              ->getResults());
      return success();
    }

    return failure();
  }
};

/// Simple greedy pattern rewrite driver for arithmetic strength reduction pass.
struct ArithReduceStrengthPass
    : public ArithReduceStrengthBase<ArithReduceStrengthPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    RewritePatternSet patterns{ctx};
    patterns.add<ReplaceMulAddWithSub>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      signalPassFailure();
  };
};
} // namespace

namespace dynamatic {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createArithReduceStrength() {
  return std::make_unique<ArithReduceStrengthPass>();
}
} // namespace dynamatic
