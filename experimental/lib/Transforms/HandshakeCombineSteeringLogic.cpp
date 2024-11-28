//===- HandshakeCombineSteeringLogic.cpp - Simplify FTD  ----*- C++ -*-----===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass which simplify the resulting FTD circuit by
// merging units which have the smae inputs and the same outputs.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/HandshakeCombineSteeringLogic.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cassert>

using namespace mlir;
using namespace dynamatic;

namespace {

/// Combine redundant init merges. These merges have one constant input and a
/// condition input. If two merges are identical, then one of them can be
/// removed
struct CombineInits : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {

    // Work only with merges having two inputs
    if (mergeOp->getNumOperands() != 2)
      return failure();

    // One of the inputs of the merge must be a constants
    int constIdx = -1;
    for (int i = 0; i < 2; i++) {
      if (isa_and_nonnull<handshake::ConstantOp>(
              mergeOp.getDataOperands()[i].getDefiningOp()))
        constIdx = i;
    }

    if (constIdx == -1)
      return failure();

    // Get the index of the other input
    int loopIdx = 1 - constIdx;

    DenseSet<handshake::MergeOp> redundantInits;
    for (auto *user : mergeOp.getDataOperands()[loopIdx].getUsers())
      if (isa_and_nonnull<handshake::MergeOp>(user) && user != mergeOp) {
        handshake::MergeOp mergeUser = cast<handshake::MergeOp>(user);
        if (isa_and_nonnull<handshake::ConstantOp>(
                mergeUser.getDataOperands()[constIdx].getDefiningOp()))
          redundantInits.insert(mergeUser);
      }

    if (redundantInits.empty())
      return failure();

    for (auto init : redundantInits) {
      rewriter.replaceAllUsesWith(init.getResult(), mergeOp.getResult());
      rewriter.eraseOp(init);
    }

    return success();
  }
};

/// Returns true if the loop under analysis is a self regenerating mux. One
/// input of the mux comes from the mux itself, while the other input comes from
/// somewhere else. If two muxes are identical in this sense, one of them can be
/// removed.
bool isSelfRegenerateMux(handshake::MuxOp muxOp, int &muxCycleInputIdx) {

  // One user must be a Branch; otherwise, the pattern match fails
  DenseSet<handshake::ConditionalBranchOp> branches;

  for (auto *muxUser : muxOp.getResult().getUsers()) {
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(muxUser)) {
      auto br = cast<handshake::ConditionalBranchOp>(muxUser);
      branches.insert(br);
    }
  }

  // The conditional branches that were found should also form a loop with the
  // multiplexer under analysis
  bool foundCycle = false;
  int operIdx = 0;
  handshake::ConditionalBranchOp condBranchOp;

  for (auto muxOperand : muxOp.getDataOperands()) {
    auto *op = muxOperand.getDefiningOp();
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(op)) {
      auto br = cast<handshake::ConditionalBranchOp>(op);
      if (branches.contains(br)) {
        foundCycle = true;
        muxCycleInputIdx = operIdx;
        condBranchOp = br;
        break;
      }
    }
    operIdx++;
  }

  return foundCycle;
}

// Note: This pattern assumes that all Muxes belonging to 1 loop have the same
// conventions about the index of the input coming from outside the loop and
// that coming from inside through a cycle
// This pattern combines all Muxes that are used to regenerate the same value
// but to different consumers.. It searches for a Mux that has a bwd edge
// (cyclic input) and searches for all Muxes using the some condition and the
// same
struct CombineMuxes : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {

    // The mux needs to have three inputs
    if (muxOp.getNumOperands() != 3)
      return failure();

    int muxCycleIdx;

    // Exit if it's not a self regenerate mux
    if (!isSelfRegenerateMux(muxOp, muxCycleIdx))
      return failure();

    int muxOutIdx = 1 - muxCycleIdx;

    DenseSet<handshake::MuxOp> dataMuxUsers;
    DenseSet<handshake::MuxOp> redundantMuxes;

    // Get users of the operation at the muxOuterInputIdx
    for (auto *dataUser : muxOp.getDataOperands()[muxOutIdx].getUsers()) {
      if (isa_and_nonnull<handshake::MuxOp>(dataUser) && dataUser != muxOp) {
        auto muxUser = cast<handshake::MuxOp>(dataUser);
        int tempValue;
        if (isSelfRegenerateMux(muxOp, tempValue))
          dataMuxUsers.insert(muxUser);
      }
    }

    // Get users of the operation at the select input, and consider only the
    // users which were also in `dataMuxUsers`
    for (auto *selUser : muxOp.getSelectOperand().getUsers())
      if (isa_and_nonnull<handshake::MuxOp>(selUser) && selUser != muxOp) {
        auto muxUser = cast<handshake::MuxOp>(selUser);
        int tempValue;
        if (isSelfRegenerateMux(muxUser, tempValue) &&
            dataMuxUsers.contains(muxUser)) {
          redundantMuxes.contains(muxUser);
        }
      }

    if (redundantMuxes.empty())
      return failure();

    // Loop over redundantMuxes and replace the users of them with the output of
    // muxOp Note that the users of all redundantMuxes include the Branches
    // forming cycles with each of them, but as we erase the redundantMuxes,
    // these Branches will have their two outputs feeding nothing and will be
    // erased using the RemoveDoubleSinkBranches
    for (auto mux : redundantMuxes) {
      rewriter.replaceAllUsesWith(mux.getResult(), muxOp.getResult());
      rewriter.eraseOp(mux);
    }

    return success();
  }
};

/// Remove conditional branches that have no successors
struct RemoveDoubleSinkBranches
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();

    // The pattern fails if the branch has either true or false successors
    if (!branchTrueResult.getUsers().empty())
      return failure();

    if (!branchFalseResult.getUsers().empty())
      return failure();

    rewriter.eraseOp(condBranchOp);
    return success();
  }
};

/// Remove branches which have the same data operands but opposite condition
/// operand
struct CombineBranchesOppositeSign
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = condBranchOp.getConditionOperand();
    DenseSet<handshake::ConditionalBranchOp> conditionBranchUsers;
    DenseSet<handshake::ConditionalBranchOp> dataBranchUsers;
    DenseSet<handshake::ConditionalBranchOp> redundantBranches;

    bool searchForANot = false;
    Value actualCondOperand = condOperand;

    if (isa_and_nonnull<handshake::NotOp>(condOperand.getDefiningOp()))
      actualCondOperand = condOperand.getDefiningOp()->getOperand(0);
    else
      searchForANot = true;

    if (!searchForANot) {
      for (auto *condUser : actualCondOperand.getUsers()) {
        if (isa_and_nonnull<handshake::ConditionalBranchOp>(condUser)) {
          auto br = cast<handshake::ConditionalBranchOp>(condUser);
          if (condUser == condBranchOp)
            continue;
          conditionBranchUsers.insert(br);
        }
      }
    } else {
      // Do not directly store all users of the condition; rather, store the
      // Branch users of any NOT that is itself a user of the condition
      for (auto *condUser : actualCondOperand.getUsers()) {
        if (!isa_and_nonnull<handshake::NotOp>(condUser))
          continue;
        handshake::NotOp notOp = cast<handshake::NotOp>(condUser);
        for (auto *notOpUser : notOp.getResult().getUsers()) {
          if (!isa_and_nonnull<handshake::ConditionalBranchOp>(notOpUser))
            continue;
          auto br = cast<handshake::ConditionalBranchOp>(notOpUser);
          conditionBranchUsers.insert(br);
        }
      }
    }

    for (auto *dataUser : dataOperand.getUsers()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(dataUser)) {
        if (dataUser == condBranchOp)
          continue;
        dataBranchUsers.insert(cast<handshake::ConditionalBranchOp>(dataUser));
      }
    }

    if (conditionBranchUsers.empty() || dataBranchUsers.empty())
      return failure();

    // Find the redundant branches
    for (auto br : dataBranchUsers)
      if (conditionBranchUsers.find(br) != conditionBranchUsers.end())
        redundantBranches.insert(br);

    if (redundantBranches.empty())
      return failure();

    // Erase the redundant branches
    for (auto br : redundantBranches) {
      rewriter.replaceAllUsesWith(br.getTrueResult(),
                                  condBranchOp.getFalseResult());
      rewriter.replaceAllUsesWith(br.getFalseResult(),
                                  condBranchOp.getTrueResult());
      rewriter.eraseOp(br);
    }

    return success();
  }
};

/// Remove branches with same data operands and same conditional operand
struct RemoveNotCondition
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    auto conditionValue = condBranchOp.getConditionOperand();
    Operation *conditionOperation = conditionValue.getDefiningOp();

    if (!llvm::isa_and_nonnull<handshake::NotOp>(conditionOperation))
      return failure();

    handshake::NotOp drivingNot =
        llvm::dyn_cast<handshake::NotOp>(conditionOperation);

    rewriter.setInsertionPointAfter(condBranchOp);

    auto newBranch = rewriter.create<handshake::ConditionalBranchOp>(
        conditionOperation->getLoc(),
        experimental::ftd::getBranchResultTypes(
            condBranchOp.getTrueResult().getType()),
        drivingNot.getOperand(), condBranchOp.getDataOperand());

    rewriter.replaceAllUsesWith(condBranchOp.getTrueResult(),
                                newBranch.getFalseResult());
    rewriter.replaceAllUsesWith(condBranchOp.getFalseResult(),
                                newBranch.getTrueResult());
    rewriter.eraseOp(condBranchOp);

    return success();
  }
};

/// Remove branches with same data operands and same conditional operand
struct CombineBranchesSameSign
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = condBranchOp.getConditionOperand();
    DenseSet<handshake::ConditionalBranchOp> conditionBranchUsers;
    DenseSet<handshake::ConditionalBranchOp> redundantBranches;

    // Get all the users of the condition operand, and keep the branches only
    for (auto *condUser : condOperand.getUsers()) {
      if (condUser == condBranchOp)
        continue;
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(condUser)) {
        auto branch = cast<handshake::ConditionalBranchOp>(condUser);
        conditionBranchUsers.insert(branch);
      }
    }

    // Check if one of the branch users of the data input was also a user of
    // the condition input: in this case, the branch is redundant
    for (auto *dataUser : dataOperand.getUsers()) {
      if (dataUser == condBranchOp)
        continue;
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(dataUser)) {
        auto branch = cast<handshake::ConditionalBranchOp>(dataUser);
        if (conditionBranchUsers.contains(branch))
          redundantBranches.insert(branch);
      }
    }

    // Nothing to erase
    if (redundantBranches.empty())
      return failure();

    // Erase the redundant branch
    for (auto br : redundantBranches) {
      rewriter.replaceAllUsesWith(br.getTrueResult(),
                                  condBranchOp.getTrueResult());
      rewriter.replaceAllUsesWith(br.getFalseResult(),
                                  condBranchOp.getFalseResult());
      rewriter.eraseOp(br);
    }
    return success();
  }
};

/// Simple driver for the Handshake Combine Branches Merges pass, based on a
/// greedy pattern rewriter.
struct HandshakeCombineSteeringLogicPass
    : public dynamatic::experimental::ftd::impl::
          HandshakeCombineSteeringLogicBase<HandshakeCombineSteeringLogicPass> {
  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveDoubleSinkBranches, CombineBranchesSameSign,
                 CombineBranchesOppositeSign, CombineInits, CombineMuxes,
                 RemoveNotCondition>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::ftd::combineSteeringLogic() {
  return std::make_unique<HandshakeCombineSteeringLogicPass>();
}
