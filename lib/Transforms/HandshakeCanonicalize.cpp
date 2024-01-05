//===-HandshakeCanonicalize.cpp - Canonicalize Handshake ops ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements rewrite patterns for the Handshake canonicalization pass, whih are
// greedily applied on the IR. These patterns do their best to attach newly
// inserted operations to known basic blocks when enough BB information is
// available. Additionally, the pass preserves the circuit's materialization
// status.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeCanonicalize.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Handshake.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace mlir;
using namespace dynamatic;

namespace {

/// Erases unconditional branches (which would eventually lower to simple
/// wires).
struct EraseUnconditionalBranches
    : public OpRewritePattern<handshake::BranchOp> {
  using OpRewritePattern<handshake::BranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::BranchOp brOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(brOp, brOp.getDataOperand());
    return success();
  }
};

/// Erases merges with a single data operand.
struct EraseSingleInputMerges : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
    if (mergeOp->getNumOperands() != 1)
      return failure();

    rewriter.replaceOp(mergeOp, mergeOp.getOperand(0));
    return success();
  }
};

/// Erases muxes with a single data operand. Inserts a sink operation to consume
/// the select operand of erased muxes.
struct EraseSingleInputMuxes : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    ValueRange dataOperands = muxOp.getDataOperands();
    if (dataOperands.size() != 1)
      return failure();

    // Insert a sink to consume the mux's select token
    rewriter.setInsertionPoint(muxOp);
    Value select = muxOp.getSelectOperand();
    rewriter.create<handshake::SinkOp>(muxOp->getLoc(), select);

    rewriter.replaceOp(muxOp, dataOperands.front());
    return success();
  }
};

/// Erases control merges with a single data operand. If necessary, inserts a
/// sourced 0 constant to replace any real uses of the index result of erased
/// control merges.
struct EraseSingleInputControlMerges
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    if (cmergeOp->getNumOperands() != 1)
      return failure();

    Value dataRes = cmergeOp.getOperand(0);
    Value indexRes = cmergeOp.getIndex();
    if (hasRealUses(indexRes)) {
      // If the index result has uses, then replace it with a sourced constant
      // with value 0 (the index of the cmerge's single input)
      rewriter.setInsertionPoint(cmergeOp);

      // Create a source operation for the constant
      handshake::SourceOp srcOp = rewriter.create<handshake::SourceOp>(
          cmergeOp->getLoc(), rewriter.getNoneType());
      inheritBB(cmergeOp, srcOp);

      /// NOTE: Sourcing this value may cause problems with very exotic uses of
      /// control merges. Ideally, we would check whether the value is sourcable
      /// first; if not we would connect the constant to the control network
      /// instead.

      // Build the attribute for the constant
      Type indexResType = indexRes.getType();
      handshake::ConstantOp cstOp = rewriter.create<handshake::ConstantOp>(
          cmergeOp.getLoc(), indexResType,
          rewriter.getIntegerAttr(indexResType, 0), srcOp.getResult());
      inheritBB(cmergeOp, cstOp);

      // Replace the cmerge's index result with a constant 0
      rewriter.replaceOp(cmergeOp, {dataRes, cstOp.getResult()});
      return success();
    }

    // Replace the cmerge's data result with its unique operand, erase any sinks
    // consuming the index result, and finally delete the cmerge
    rewriter.replaceAllUsesWith(cmergeOp.getResult(), dataRes);
    eraseSinkUsers(indexRes, rewriter);
    rewriter.eraseOp(cmergeOp);
    return success();
  }
};

/// Downgrades control merges whose index result has no real uses to simpler
/// yet equivalent merges.
struct DowngradeIndexlessControlMerge
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    Value indexRes = cmergeOp.getIndex();
    if (hasRealUses(indexRes))
      return failure();

    // Create a merge operation to replace the cmerge
    rewriter.setInsertionPoint(cmergeOp);
    handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(
        cmergeOp.getLoc(), cmergeOp->getOperands());
    inheritBB(cmergeOp, mergeOp);

    // Replace the cmerge's data result with the merge's result, erase any
    // sinks consuming the index result, and finally delete the cmerge
    rewriter.replaceAllUsesWith(cmergeOp.getResult(), mergeOp.getResult());
    eraseSinkUsers(indexRes, rewriter);
    rewriter.eraseOp(cmergeOp);
    return success();
  }
};

/// Eliminates forks feeding into other forks by replacing both with a single
/// fork operation.
struct EliminateForksToForks : OpRewritePattern<handshake::ForkOp> {
  using mlir::OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // The defining operation must be also be a fork for the pattern to apply
    Value forkOprd = forkOp.getOperand();
    auto defForkOp = forkOprd.getDefiningOp<handshake::ForkOp>();
    if (!defForkOp)
      return failure();

    // It is important to take into account whether the matched fork's operand
    // is the single use of the defining fork's corresponding result. If it is
    // not, the new combined fork needs an extra result to replace the defining
    // fork's result with in the other uses
    bool isForkOprdSingleUse = forkOprd.hasOneUse();

    // Create a new combined fork to replace the two others
    unsigned totalNumResults = forkOp.getSize() + defForkOp.getSize();
    if (isForkOprdSingleUse)
      --totalNumResults;
    rewriter.setInsertionPoint(defForkOp);
    handshake::ForkOp newForkOp = rewriter.create<handshake::ForkOp>(
        defForkOp.getLoc(), defForkOp.getOperand(), totalNumResults);
    inheritBB(defForkOp, newForkOp);

    // Replace the defining fork's results with the first results of the new
    // fork (skipping the result feeding the matched fork if it has a single
    // use)
    ValueRange newResults = newForkOp->getResults();
    auto newResIt = newResults.begin();
    for (OpResult defForkRes : defForkOp->getResults()) {
      if (!isForkOprdSingleUse || defForkRes != forkOprd)
        rewriter.replaceAllUsesWith(defForkRes, *(newResIt++));
    }

    // Replace the results of the matched fork with the corresponding results of
    // the new defining fork
    rewriter.replaceOp(forkOp, newResults.take_back(forkOp.getSize()));
    return success();
  }
};

/// Erases forks with a single result unless their operand originates from a
/// lazy fork, in which case they may exist to prevent a combinational cycle.
struct EraseSingleOutputForks : OpRewritePattern<handshake::ForkOp> {
  using mlir::OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // The fork must have a single result
    if (forkOp.getSize() != 1)
      return failure();

    // The defining operation must not be a lazy fork, otherwise the fork may be
    // here to avoid a combination cycle between the valid and ready wires
    if (forkOp.getOperand().getDefiningOp<handshake::LazyForkOp>())
      return failure();

    // Bypass the fork and succeed
    rewriter.replaceOp(forkOp, forkOp.getOperand());
    return success();
  }
};

/// Removes outputs of forks that do not have real uses. This can result in the
/// size reduction or deletion of fork operations (the latter if none of the
/// fork results have real users) as well as sink users of fork results.
struct MinimizeForkSizes : OpRewritePattern<handshake::ForkOp> {
  using OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // Compute the list of fork results that are actually used (erase any sink
    // user along the way)
    SmallVector<Value> usedForkResults;
    for (OpResult res : forkOp.getResults()) {
      if (hasRealUses(res)) {
        usedForkResults.push_back(res);
      } else if (!res.use_empty()) {
        // The value has sink users, delete them as the fork producing their
        // operand will be removed
        for (Operation *sinkUser : llvm::make_early_inc_range(res.getUsers()))
          rewriter.eraseOp(sinkUser);
      }
    }
    // Fail if all fork results are used, since it means that no transformation
    // is requires
    if (usedForkResults.size() == forkOp->getNumResults())
      return failure();

    if (!usedForkResults.empty()) {
      // Create a new fork operation
      rewriter.setInsertionPoint(forkOp);
      handshake::ForkOp newForkOp = rewriter.create<handshake::ForkOp>(
          forkOp.getLoc(), forkOp.getOperand(), usedForkResults.size());
      inheritBB(forkOp, newForkOp);

      // Replace results with actual uses of the original fork with results from
      // the new fork
      ValueRange newResults = newForkOp.getResult();
      for (auto [oldRes, newRes] : llvm::zip(usedForkResults, newResults))
        rewriter.replaceAllUsesWith(oldRes, newRes);
    }
    rewriter.eraseOp(forkOp);
    return success();
  }
};

struct DoNotForkConstants : public OpRewritePattern<handshake::ForkOp> {
  using OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // The fork must be fed by a constant, possibly extended/truncated
    Operation *defOp = forkOp.getOperand().getDefiningOp();
    SmallVector<Operation *> bitMods;
    while (isa_and_nonnull<arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(
        defOp)) {
      bitMods.push_back(defOp);
      defOp = defOp->getOperand(0).getDefiningOp();
    }
    if (!defOp || !isa<handshake::ConstantOp>(defOp))
      return failure();
    handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(defOp);
    mlir::TypedAttr cstVal = cstOp.getValue();

    // Create as many constants (+ possible extensions/truncations) as there are
    // fork outputs, and create a new fork for the control signal
    auto newForkOp = rewriter.create<handshake::ForkOp>(
        cstOp->getLoc(), cstOp.getOperand(), forkOp.getNumResults());
    inheritBB(cstOp, newForkOp);
    SmallVector<Value> newResults;
    for (OpResult ctrlOpr : newForkOp->getResults()) {
      // Create the new constant
      auto newCstOp = rewriter.create<handshake::ConstantOp>(
          cstOp->getLoc(), cstVal.getType(), cstVal, ctrlOpr);
      Value cstRes = newCstOp.getResult();
      inheritBB(cstOp, newCstOp);

      // Recreate bitwidth modifiers in the same order (iterate in reverse
      // discovery order)
      for (size_t idx = bitMods.size(); idx >= 1; --idx) {
        Operation *mod = bitMods[idx - 1];
        Operation *newMod = rewriter.create(
            mod->getLoc(),
            StringAttr::get(getContext(), mod->getName().getStringRef()),
            {cstRes}, mod->getResultTypes());
        inheritBB(mod, newMod);
        cstRes = newMod->getResult(0);
      }
      newResults.push_back(cstRes);
    }

    rewriter.replaceOp(forkOp, newResults);
    return success();
  }
};

/// Simple driver for the Handshake canonicalization pass, based on a greedy
/// pattern rewriter.
struct HandshakeCanonicalizePass
    : public dynamatic::impl::HandshakeCanonicalizeBase<
          HandshakeCanonicalizePass> {

  HandshakeCanonicalizePass(bool justBranches) {
    this->justBranches = justBranches;
  }

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    mlir::ModuleOp mod = getOperation();

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns
        .add<EraseUnconditionalBranches, DoNotForkConstants,
             EliminateForksToForks, EraseSingleOutputForks, MinimizeForkSizes>(
            ctx);
    if (!justBranches)
      patterns.add<EraseSingleInputMerges, EraseSingleInputMuxes,
                   EraseSingleInputControlMerges>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeCanonicalize(bool justBranches) {
  return std::make_unique<HandshakeCanonicalizePass>(justBranches);
}
