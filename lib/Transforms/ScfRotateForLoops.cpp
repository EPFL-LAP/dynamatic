//===- ScfRotateForLoops.cpp - Rotate for loops into do-while's -*- C++ -*-===//
//
// Implements the --scf-rotate-for-loops pass, which tranforms for loops
// provably executing at least once into equivalent do-while loops.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/ScfRotateForLoops.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace dynamatic;
using namespace circt::handshake;

/// Determines whether a for loop is valid for rotation i.e., whether we can
/// determine that it will execute at least once. The heuristic implemented by
/// this function is necessarily conservative.
static bool isLegalForRotation(scf::ForOp forOp) {
  // Check that both bound values are defined by operations
  auto lbDef = forOp.getLowerBound().getDefiningOp();
  auto ubDef = forOp.getUpperBound().getDefiningOp();
  if (!lbDef || !ubDef)
    return false;

  // Check that both bounds of the for loop are statically known and that the
  // lower bound is strictly less than the upper bound
  if (auto lbCst = dyn_cast<arith::ConstantOp>(lbDef)) {
    if (auto ubCst = dyn_cast<arith::ConstantOp>(ubDef)) {
      auto lbVal = dyn_cast<IntegerAttr>(lbCst.getValue());
      auto ubVal = dyn_cast<IntegerAttr>(ubCst.getValue());
      if (!lbVal || !ubVal)
        return false;
      return lbVal.getValue().getZExtValue() < ubVal.getValue().getZExtValue();
    }
  }

  return false;
}

namespace {

struct RotateLoop : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!isLegalForRotation(forOp))
      return failure();

    rewriter.setInsertionPoint(forOp);

    // Loop-carried arguments to the do-while loop are the same as for the for
    // loop, with the explicit addition of the IV
    SmallVector<Value> whileOpArgs;
    whileOpArgs.push_back(forOp.getLowerBound());
    llvm::copy(forOp.getInitArgs(), std::back_inserter(whileOpArgs));

    // Create a do-while that is equivalent to the loop
    ValueRange whileArgsRange(whileOpArgs);
    auto whileOp =
        rewriter.create<scf::WhileOp>(forOp.getLoc(), whileArgsRange.getTypes(),
                                      whileOpArgs, nullptr, nullptr);

    // Move all operations from the for loop body to the "before" region of
    // the while loop
    Block &beforeBlock = whileOp.getBefore().front();
    rewriter.mergeBlocks(&forOp.getRegion().front(), &beforeBlock,
                         beforeBlock.getArguments());

    // Check the for loop condition at the end of the before block
    rewriter.setInsertionPointToEnd(&beforeBlock);
    auto addOp = rewriter.create<arith::AddIOp>(
        forOp->getLoc(), beforeBlock.getArguments().front(), forOp.getStep());
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        forOp->getLoc(), arith::CmpIPredicate::ult, addOp.getResult(),
        forOp.getUpperBound());

    // Identify the yield operation that was moved from the for loop body to
    // the before block
    auto yields = beforeBlock.getOps<scf::YieldOp>();
    assert(!yields.empty() && "no yields moved from for to while loop");
    assert(++yields.begin() == yields.end() && "expected only one yield");
    auto yieldOp = *yields.begin();

    // Replace the for loop yield terminator with a while condition terminator
    SmallVector<Value> condOperands;
    condOperands.push_back(addOp.getResult());
    for (auto op : yieldOp->getOperands())
      condOperands.push_back(op);
    auto condOp = rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        yieldOp, cmpOp.getResult(), condOperands);

    // The after block of the while loop is a simple forwarder in case of
    // do-while loops, we just need to yield all loop-carried values back to
    // the before block
    Block &afterBlock = whileOp.getAfter().front();
    rewriter.setInsertionPointToStart(&afterBlock);
    rewriter.create<scf::YieldOp>(condOp->getLoc(), afterBlock.getArguments());

    // Replace for's results with while's results (drop while's first
    // result, which is the IV)
    for (auto res :
         llvm::zip(forOp->getResults(), whileOp.getResults().drop_front()))
      std::get<0>(res).replaceAllUsesWith(std::get<1>(res));
    return success();
  }
};

/// Simple greedy pattern rewrite driver for SCF loop rotation pass.
struct ScfForLoopRotationPass
    : public ScfForLoopRotationBase<ScfForLoopRotationPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    RewritePatternSet patterns{ctx};
    patterns.add<RotateLoop>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      signalPassFailure();
  };
};
} // namespace

namespace dynamatic {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createScfRotateForLoops() {
  return std::make_unique<ScfForLoopRotationPass>();
}
} // namespace dynamatic
