//===- HandshakePrepareForLegacy.h - Prepare for legacy flow ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a preprocessing step for handshake-level IR to make it
// compatible with the legacy Dynamatic flow (through DOT export).
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakePrepareForLegacy.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

/// Creates a corresponding conditional branch for each unconditional branch.
/// The data input of corresponding branches is the same. A constant true
/// triggered by the given control signal is created to feed the conditional
/// branches' condition input.
static void createNewBranches(ArrayRef<handshake::BranchOp> branches,
                              Value ctrl, OpBuilder &builder) {
  // Create constant source of true conditions
  builder.setInsertionPointAfterValue(ctrl);
  IntegerAttr cond = builder.getBoolAttr(true);
  auto constOp = builder.create<handshake::ConstantOp>(
      ctrl.getLoc(), cond.getType(), cond, ctrl);

  // Try to set the bb attribute on the merge
  if (auto defOp = ctrl.getDefiningOp())
    inheritBB(defOp, constOp);
  else
    constOp->setAttr(BB_ATTR, builder.getUI32IntegerAttr(0));

  // Insert a conditional branch for every unconditional branch and replace the
  // latter's result uses with the "true" result of the former
  auto trueCond = constOp.getResult();
  for (auto br : branches) {
    builder.setInsertionPointAfter(br);
    auto cbranch = builder.create<handshake::ConditionalBranchOp>(
        br.getLoc(), trueCond, br->getOperand(0));
    inheritBB(br, cbranch);
    br.getResult().replaceAllUsesWith(cbranch.getTrueResult());
  }
}

/// Converts all unconditional branches of a function into conditional branches
/// with a constant true condition input.
static void convertBranches(handshake::FuncOp funcOp, OpBuilder &builder) {
  auto branches = funcOp.getOps<handshake::BranchOp>();
  if (branches.empty())
    return;

  auto handshakeBlocks = getLogicBBs(funcOp);

  // Iterate over all identified handshake blocks to identify unconditional
  // branches and convert them
  SmallVector<handshake::BranchOp> branchesOutOfBlocks;
  for (auto &[blockID, blockOps] : handshakeBlocks.blocks) {

    // Identify all unconditional branches in the block, as well as a control
    // merge with dataless inputs if possible
    SmallVector<handshake::BranchOp> blockBranchOps;
    handshake::ControlMergeOp cmerge = nullptr;
    for (auto op : blockOps)
      if (isa<handshake::BranchOp>(op))
        blockBranchOps.push_back(dyn_cast<handshake::BranchOp>(op));
      else if (isa<handshake::ControlMergeOp>(op)) {
        auto blockCMerge = dyn_cast<handshake::ControlMergeOp>(op);
        if (!cmerge && blockCMerge.isControl())
          cmerge = blockCMerge;
      }

    if (blockBranchOps.empty())
      continue;

    if (cmerge)
      // If we found a control merge with dataless inputs in the block, use its
      // result as control value for the new conditional branches
      createNewBranches(blockBranchOps, cmerge.getResult(), builder);
    else if (blockID == 0) {
      // If we are in the entry block, we can use the start input of the
      // function (last argument) as our control value
      assert(funcOp.getArguments().back().getType().isa<NoneType>() &&
             "expected last function argument to be a NoneType");
      createNewBranches(blockBranchOps, funcOp.getArguments().back(), builder);
    } else
      // If we did not find a control merge with dataless inputs in the
      // block, we'll simply create an endless source of true conditions
      // outside of all blocks to trigger the conditional branches
      llvm::copy(blockBranchOps, std::back_inserter(branchesOutOfBlocks));
  }

  // Collect all unconditional branches that are out of blocks
  for (auto op : handshakeBlocks.outOfBlocks)
    if (isa<handshake::BranchOp>(op))
      branchesOutOfBlocks.push_back(dyn_cast<handshake::BranchOp>(op));

  if (!branchesOutOfBlocks.empty()) {
    // Create an endless source of control signals at the beginning of the
    // function for all unconditional branches that could not be matched with a
    // dataless control merge or the function's start control
    builder.setInsertionPointToStart(&funcOp.front());
    Value ctrl = builder
                     .create<handshake::SourceOp>(
                         funcOp.front().front().getLoc(), builder.getNoneType())
                     .getResult();
    createNewBranches(branchesOutOfBlocks, ctrl, builder);
  }

  // Delete all unconditional branches
  for (auto brOp : llvm::make_early_inc_range(branches))
    brOp->erase();
}

namespace {

/// Simplifies control merges into simple merges when they only have one operand
/// or when their index result is unused.
struct SimplifyCMerges : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    // Only operate on control merges part of the control network
    if (!cmergeOp.getResult().getType().isa<NoneType>())
      return failure();

    auto numOperands = cmergeOp->getNumOperands();
    auto indexUnused = cmergeOp.getIndex().use_empty();
    if (numOperands != 1 && !indexUnused)
      return failure();

    // A cmerge with one operands or an unused index result can be downgraded
    // to a simple merge with the same operands
    rewriter.setInsertionPoint(cmergeOp);
    auto mergeOp = rewriter.create<handshake::MergeOp>(
        cmergeOp.getLoc(), cmergeOp.getDataOperands());
    inheritBB(cmergeOp, mergeOp);
    rewriter.updateRootInPlace(cmergeOp, [&] {
      cmergeOp.getResult().replaceAllUsesWith(mergeOp.getResult());

      if (indexUnused)
        return;
      // When the index has users, we must replace it with a constant 0
      // triggered by the new merge

      // Create the attribute for the constant, whose type is derived from the
      // cmerge's index result (index or integer attribute)
      auto indexResType = cmergeOp.getIndex().getType();
      TypedAttr constantAttr;
      if (isa<IndexType>(indexResType))
        constantAttr = rewriter.getIndexAttr(0);
      else
        constantAttr = rewriter.getIntegerAttr(indexResType, 0);

      // Create the constant and replace the cmerge's index result
      auto constantOp = rewriter.create<handshake::ConstantOp>(
          cmergeOp.getLoc(), constantAttr.getType(), constantAttr,
          mergeOp.getResult());
      inheritBB(cmergeOp, constantOp);
      cmergeOp.getIndex().replaceAllUsesWith(constantOp.getResult());
    });
    rewriter.eraseOp(cmergeOp);

    return success();
  }
};

/// Erase single-input merges in the entry block of a function to match the
/// behavior of legacy Dynamatic.
struct EraseEntryBlockMerges : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
    if (mergeOp.getNumOperands() > 1)
      return failure();

    if (auto bbAttr = mergeOp->getAttrOfType<IntegerAttr>(BB_ATTR))
      if (bbAttr.getValue().getZExtValue() == 0) {
        // Merge belongs to the entry block
        rewriter.replaceAllUsesWith(mergeOp.getResult(), mergeOp.getOperands());
        rewriter.eraseOp(mergeOp);
        return success();
      }

    return failure();
  }
};

/// Simple driver for prepare for legacy pass.
struct HandshakePrepareForLegacyPass
    : public HandshakePrepareForLegacyBase<HandshakePrepareForLegacyPass> {

  void runDynamaticPass() override {
    auto *ctx = &getContext();
    OpBuilder builder{ctx};

    // Convert all unconditional branches to "fake" conditional branches
    for (auto funcOp : getOperation().getOps<handshake::FuncOp>())
      convertBranches(funcOp, builder);

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    RewritePatternSet patterns{ctx};
    patterns.add<SimplifyCMerges, EraseEntryBlockMerges>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakePrepareForLegacy() {
  return std::make_unique<HandshakePrepareForLegacyPass>();
}
