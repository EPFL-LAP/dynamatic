//===- ScfRotateForLoops.cpp - Rotate for loops into do-while's -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --scf-rotate-for-loops pass, which tranforms for loops
// provably executing at least once into equivalent do-while loops.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/ScfRotateForLoops.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NumericAnalysis.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace dynamatic;
using namespace circt::handshake;

namespace {

struct RotateLoop : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    arith::CmpIPredicate pred;
    if (!isLegalForRotation(forOp, pred))
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

    // Move all operations from the for loop body to the "before" region of the
    // while loop
    Block &beforeBlock = whileOp.getBefore().front();
    rewriter.mergeBlocks(&forOp.getRegion().front(), &beforeBlock,
                         beforeBlock.getArguments());

    // Check the for loop condition at the end of the before block
    rewriter.setInsertionPointToEnd(&beforeBlock);
    auto addOp = rewriter.create<arith::AddIOp>(
        forOp->getLoc(), beforeBlock.getArguments().front(), forOp.getStep());
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        forOp->getLoc(), pred, addOp.getResult(), forOp.getUpperBound());

    // Get the yield operation that was moved from the for loop body to the
    // before block
    scf::YieldOp yieldOp = *beforeBlock.getOps<scf::YieldOp>().begin();
    assert(yieldOp && "expected to find a yield");

    // Replace the for loop yield terminator with a while condition terminator
    SmallVector<Value> condOperands;
    condOperands.push_back(addOp.getResult());
    llvm::copy(yieldOp->getOperands(), std::back_inserter(condOperands));
    auto condOp = rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        yieldOp, cmpOp.getResult(), condOperands);

    // The after block of the while loop is a simple forwarder in case of
    // do-while loops, we just need to yield all loop-carried values back to
    // the before block
    Block &afterBlock = whileOp.getAfter().front();
    rewriter.setInsertionPointToStart(&afterBlock);
    rewriter.create<scf::YieldOp>(condOp->getLoc(), afterBlock.getArguments());

    // Replace for's results with while's results (drop while's first result,
    // which is the IV)
    rewriter.replaceOp(forOp, whileOp.getResults().drop_front());
    return success();
  }

private:
  /// Determines whether a for loop is valid for rotation i.e., whether we can
  /// determine that it will execute at least once. The heuristic implemented by
  /// this function is necessarily conservative. If the function returns true,
  /// pred contains the comparison predicate to use to evaluate the condition of
  /// the to-be-created do-while loop; otherwise its value is undefined.
  bool isLegalForRotation(scf::ForOp forOp, arith::CmpIPredicate &pred) const;
};
} // namespace

bool RotateLoop::isLegalForRotation(scf::ForOp forOp,
                                    arith::CmpIPredicate &pred) const {
  NumericAnalysis analysis;

  // Get the ranges
  NumericRange lbRange = analysis.getRange(forOp.getLowerBound());
  NumericRange ubRange = analysis.getRange(forOp.getUpperBound());

  // Check whether the loop will execute at least once
  if (!(lbRange < ubRange))
    return false;

  // Determine comparison predicate to use when rotating the loop. We can
  // insert an unsigned comparison only if the lower bound added to the
  // (guaranteed positive) step can be guaranteed to be non-negative, since
  // the first comparison will occur after the first iteration of the old for
  // loop body / new do-while body
  NumericRange stepRange = analysis.getRange(forOp.getStep());
  pred = NumericRange::add(lbRange, stepRange).isPositive()
             ? arith::CmpIPredicate::ult
             : arith::CmpIPredicate::slt;
  return true;
}

namespace {

/// Simple greedy pattern rewrite driver for SCF loop rotation pass.
struct ScfForLoopRotationPass
    : public ScfForLoopRotationBase<ScfForLoopRotationPass> {

  void runDynamaticPass() override {
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
std::unique_ptr<dynamatic::DynamaticPass> createScfRotateForLoops() {
  return std::make_unique<ScfForLoopRotationPass>();
}
} // namespace dynamatic
