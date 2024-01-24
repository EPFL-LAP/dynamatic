//===- GreedySimplifyMergeLike.cpp - Simplifies merge-like ops --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --tutorial-handshake-greedy-simplify-merge-like pass, which
// uses a greedy pattern rewriter to modify the IR within each handshake
// function.
//
//===----------------------------------------------------------------------===//

#include "tutorials/CreatingPasses/Transforms/GreedySimplifyMergeLike.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace dynamatic;

namespace {

/// Rewrite pattern that matches on all merge operations and erases those with a
/// single operand.
struct EraseSingleInputMerge : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {

    // We can ony perform the transform if the merge operation has a single
    // operand
    if (mergeOp->getNumOperands() != 1)
      return failure();

    rewriter.updateRootInPlace(mergeOp, [&] {
      // Replace all occurences of the merge's single result throughout the IR
      // with the merge's single operand. This is equivalent to bypassing the
      // merge
      rewriter.replaceAllUsesWith(mergeOp.getResult(), mergeOp.getOperand(0));

      // Erase the merge operation, whose result now has no uses
      rewriter.eraseOp(mergeOp);
    });

    // Return a success to indicate that the pattern successfully matched a
    // merge that was transfomed
    return success();
  }
};

/// Rewrite pattern that matches on all control merge operations and downgrades
/// those whose index result is not used to simpler merge operations.
struct DowngradeIndexlessControlMerge
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    // Get the control merge's index result (second result).
    // Equivalently, we could have written:
    //  auto indexResult = cmergeOp->getResult(1);
    // but using getIndex() is more readable and maintainable
    Value indexResult = cmergeOp.getIndex();

    // We can only perform the transformation if the control merge operation's
    // index result is not used throughout the IR
    if (!indexResult.use_empty())
      return failure();

    // Now, we create a new merge operation at the same position in the IR as
    // the control merge we are replacing. The merge has the exact same inputs
    // as the control merge
    rewriter.setInsertionPoint(cmergeOp);
    handshake::MergeOp newMergeOp = rewriter.create<handshake::MergeOp>(
        cmergeOp.getLoc(), cmergeOp->getOperands());

    // We are modifying the operation
    rewriter.updateRootInPlace(cmergeOp, [&] {
      // Then, replace the control merge's first result (the selected input)
      // with the single result of the newly created merge operation
      Value mergeRes = newMergeOp.getResult();
      rewriter.replaceAllUsesWith(cmergeOp.getResult(), mergeRes);

      // Finally, we can delete the original control merge, whose results have
      // no uses anymore
      rewriter.eraseOp(cmergeOp);
    });

    // Return a success to indicate that the pattern successfully matched a
    // control merge that was transfomed
    return success();
  }
};

/// Simple pass driver for our merge-like simplification transformation that
/// uses a greedy pattern rewriter internally. It will recursively try to apply
/// our two rewrite patterns on all operations within the module, including
/// nested ones.
struct GreedySimplifyMergeLikePass
    : public dynamatic::tutorials::impl::GreedySimplifyMergeLikeBase<
          GreedySimplifyMergeLikePass> {

  void runOnOperation() override {
    // Get the MLIR context for the current operation being transformed
    MLIRContext *ctx = &getContext();
    // Get the operation being transformed (the top level module)
    ModuleOp mod = getOperation();

    // Set up a configuration object to customize the behavior of the rewriter
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    // Create a rewrite pattern set and add our two patterns to it
    RewritePatternSet patterns{ctx};
    patterns.add<EraseSingleInputMerge, DowngradeIndexlessControlMerge>(ctx);

    // Apply our two patterns recursively on all operations in the input module
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace

namespace dynamatic {
namespace tutorials {

/// Returns a unique pointer to an operation pass that matches MLIR modules.
/// In our case, this is simply an instance of our unparameterized
/// GreedySimplifyMergeLikePass driver.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createGreedySimplifyMergeLikePass() {
  return std::make_unique<GreedySimplifyMergeLikePass>();
}
} // namespace tutorials
} // namespace dynamatic
