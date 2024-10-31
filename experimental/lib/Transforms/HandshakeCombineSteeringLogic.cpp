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
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <cassert>

using namespace mlir;
using namespace dynamatic;

namespace {
struct CombineInits : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
    // Doublecheck that the Merge has 2 inputs
    if (mergeOp->getNumOperands() != 2)
      return success();
    // One of the inputs of the Merge must be a constant; otherwise the pattern
    // match fails
    if (!isa_and_nonnull<handshake::ConstantOp>(
            mergeOp.getDataOperands()[0].getDefiningOp()) &&
        !isa_and_nonnull<handshake::ConstantOp>(
            mergeOp.getDataOperands()[1].getDefiningOp()))
      return success();
    int constOperandIdx;
    if (isa_and_nonnull<handshake::ConstantOp>(
            mergeOp.getDataOperands()[0].getDefiningOp()))
      constOperandIdx = 0;
    else {
      assert(isa_and_nonnull<handshake::ConstantOp>(
          mergeOp.getDataOperands()[1].getDefiningOp()));
      constOperandIdx = 1;
    }
    // auto constVal = cast<handshake::ConstantOp>(constOperandIdx).getValue();
    int loopCondOperandIdx = (constOperandIdx == 1) ? 0 : 1;
    DenseSet<handshake::MergeOp> redundantInits;
    for (auto *user : mergeOp.getDataOperands()[loopCondOperandIdx].getUsers())
      if (isa_and_nonnull<handshake::MergeOp>(user) && user != mergeOp) {
        handshake::MergeOp mergeUser = cast<handshake::MergeOp>(user);
        if (isa_and_nonnull<handshake::ConstantOp>(
                mergeUser.getDataOperands()[constOperandIdx].getDefiningOp()))
          redundantInits.insert(mergeUser);
      }
    if (redundantInits.empty())
      return failure();
    for (auto init : redundantInits) {
      handshake::MergeOp redunInit = cast<handshake::MergeOp>(init);
      rewriter.replaceAllUsesWith(redunInit.getResult(), mergeOp.getResult());
      rewriter.eraseOp(init);
    }
    llvm::dbgs() << "\t***Combine INITs***\n";
    return success();
  }
};
bool isSelfRegenerateMux(handshake::MuxOp muxOp, int &muxOuterInputIdx,
                         int &muxCycleInputIdx) {
  // One user must be a Branch; otherwise, the pattern match fails
  bool foundCondBranch = false;
  DenseSet<handshake::ConditionalBranchOp> branches;
  for (auto *muxUser : muxOp.getResult().getUsers()) {
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(muxUser)) {
      foundCondBranch = true;
      branches.insert(cast<handshake::ConditionalBranchOp>(muxUser));
    }
  }
  // This condBranchOp must also be an operand forming a cycle with the
  // mux; otherwise, the pattern match fails
  bool foundCycle = false;
  int operIdx = 0;
  handshake::ConditionalBranchOp condBranchOp;
  for (auto muxOperand : muxOp.getDataOperands()) {
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(
            muxOperand.getDefiningOp()))
      if (branches.contains(cast<handshake::ConditionalBranchOp>(
              muxOperand.getDefiningOp()))) {
        foundCycle = true;
        muxCycleInputIdx = operIdx;
        condBranchOp =
            cast<handshake::ConditionalBranchOp>(muxOperand.getDefiningOp());
        break;
      }
    operIdx++;
  }
  if (foundCondBranch)
    muxOuterInputIdx = !muxCycleInputIdx;
  return (foundCycle);
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
    // Doublecheck that the Mux has 3 inputs
    if (muxOp.getNumOperands() != 3)
      return success();
    int muxOuterInputIdx = -1;
    int muxCycleInputIdx = -1;
    if (!isSelfRegenerateMux(muxOp, muxOuterInputIdx, muxCycleInputIdx))
      return success();
    DenseSet<handshake::MuxOp> conditionMuxUsers;
    DenseSet<handshake::MuxOp> dataMuxUsers;
    DenseSet<handshake::MuxOp> redundantMuxes;
    // Get users of the operation at the muxOuterInputIdx
    for (auto *dataUser : muxOp.getDataOperands()[muxOuterInputIdx].getUsers())
      if (isa_and_nonnull<handshake::MuxOp>(dataUser) && dataUser != muxOp) {
        int tempMuxouterInputIdx = -1;
        int tempMuxCycleInputIdx = -1;
        if (isSelfRegenerateMux(cast<handshake::MuxOp>(dataUser),
                                tempMuxouterInputIdx, tempMuxCycleInputIdx))
          dataMuxUsers.insert(cast<handshake::MuxOp>(dataUser));
      }
    // Get users of the operation at the select (condition) of the Mux
    for (auto *selUser : muxOp.getSelectOperand().getUsers())
      if (isa_and_nonnull<handshake::MuxOp>(selUser) && selUser != muxOp) {
        int tempMuxouterInputIdx = -1;
        int tempMuxCycleInputIdx = -1;
        if (isSelfRegenerateMux(cast<handshake::MuxOp>(selUser),
                                tempMuxouterInputIdx, tempMuxCycleInputIdx))
          conditionMuxUsers.insert(cast<handshake::MuxOp>(selUser));
      }
    // Loop over dataMuxUsers and consider those that are found in
    // muxUsers
    for (auto mux : dataMuxUsers)
      if (conditionMuxUsers.find(mux) != conditionMuxUsers.end())
        redundantMuxes.insert(mux);
    if (redundantMuxes.empty())
      return success();
    // Loop over redundantMuxes and replace the users of them with the output of
    // muxOp Note that the users of all redundantMuxes include the Branches
    // forming cycles with each of them, but as we erase the redundantMuxes,
    // these Branches will have their two outputs feeding nothing and will be
    // erased using the RemoveDoubleSinkeBranches
    for (auto mux : redundantMuxes) {
      handshake::MuxOp redunMux = cast<handshake::MuxOp>(mux);
      rewriter.replaceAllUsesWith(redunMux.getResult(), muxOp.getResult());
      rewriter.eraseOp(mux);
    }
    llvm::dbgs() << "\t***Combine Muxes***\n";
    return success();
  }
};
/// Remove Conditional Branches that have no successors
struct RemoveDoubleSinkBranches
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();
    // Pattern match fails if the Branch has a true or false successor
    if (!branchTrueResult.getUsers().empty() ||
        !branchFalseResult.getUsers().empty())
      return success();
    rewriter.eraseOp(condBranchOp);
    return success();
  }
};
struct CombineBranchesOppositeSign
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    // 1st step: Get the input data value and input condition value
    // 2nd step: Get the users of both the data value and condition value, and
    // if there are common users that are of type Branch, they should be all
    // combined to a single Branch
    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = condBranchOp.getConditionOperand();
    DenseSet<handshake::ConditionalBranchOp> conditionBranchUsers;
    DenseSet<handshake::ConditionalBranchOp> dataBranchUsers;
    DenseSet<handshake::ConditionalBranchOp> redundantBranches;
    // If the condition of condBranchOp is a NOT, then we need to get its input
    bool searchForANot = false;
    Value actualCondOperand = condOperand;
    if (isa_and_nonnull<handshake::NotOp>(condOperand.getDefiningOp()))
      actualCondOperand = condOperand.getDefiningOp()->getOperand(0);
    else
      searchForANot = true;
    if (!searchForANot) {
      for (auto *condUser : actualCondOperand.getUsers())
        if (isa_and_nonnull<handshake::ConditionalBranchOp>(condUser) &&
            condUser != condBranchOp)
          conditionBranchUsers.insert(
              cast<handshake::ConditionalBranchOp>(condUser));
    } else {
      // Do not directly store all users of the condition; rather, store the
      // Branch users of any NOT that is itself a user of the condition
      for (auto *condUser : actualCondOperand.getUsers())
        if (isa_and_nonnull<handshake::NotOp>(condUser)) {
          handshake::NotOp notOp = cast<handshake::NotOp>(condUser);
          for (auto *notOpUser : notOp.getResult().getUsers()) {
            if (isa_and_nonnull<handshake::ConditionalBranchOp>(notOpUser))
              conditionBranchUsers.insert(
                  cast<handshake::ConditionalBranchOp>(notOpUser));
          }
        }
    }
    for (auto *dataUser : dataOperand.getUsers())
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(dataUser) &&
          dataUser != condBranchOp)
        dataBranchUsers.insert(cast<handshake::ConditionalBranchOp>(dataUser));
    if (conditionBranchUsers.empty() || dataBranchUsers.empty())
      return success();
    // Loop over dataBranchUsers and consider those that are found in
    // conditionBranchUsers
    for (auto br : dataBranchUsers)
      if (conditionBranchUsers.find(br) != conditionBranchUsers.end())
        redundantBranches.insert(br);
    if (redundantBranches.empty())
      return success();
    // Erase redundant Branches by putting their true succs at the false succs
    // of the condBranchOp and their false succs at the true succs of the
    // condBranchOp
    for (auto br : redundantBranches) {
      handshake::ConditionalBranchOp redunBr =
          cast<handshake::ConditionalBranchOp>(br);
      rewriter.replaceAllUsesWith(redunBr.getTrueResult(),
                                  condBranchOp.getFalseResult());
      rewriter.replaceAllUsesWith(redunBr.getFalseResult(),
                                  condBranchOp.getTrueResult());
      rewriter.eraseOp(br);
    }
    llvm::dbgs() << "\t***Combine Branches Oppostie Signs***\n";
    return success();
  }
};
struct CombineBranchesSameSign
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    // 1st step: Get the input data value and input condition value
    // 2nd step: Get the users of both the data value and condition value, and
    // if there are common users that are of type Branch, they should be all
    // combined to a single Branch
    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = condBranchOp.getConditionOperand();
    DenseSet<handshake::ConditionalBranchOp> conditionBranchUsers;
    DenseSet<handshake::ConditionalBranchOp> dataBranchUsers;
    DenseSet<handshake::ConditionalBranchOp> redundantBranches;
    for (auto *condUser : condOperand.getUsers())
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(condUser) &&
          condUser != condBranchOp)
        conditionBranchUsers.insert(
            cast<handshake::ConditionalBranchOp>(condUser));
    for (auto *dataUser : dataOperand.getUsers())
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(dataUser) &&
          dataUser != condBranchOp)
        dataBranchUsers.insert(cast<handshake::ConditionalBranchOp>(dataUser));
    if (conditionBranchUsers.empty() || dataBranchUsers.empty())
      return success();
    // Loop over dataBranchUsers and consider those that are found in
    // conditionBranchUsers
    for (auto br : dataBranchUsers)
      if (conditionBranchUsers.find(br) != conditionBranchUsers.end())
        redundantBranches.insert(br);
    if (redundantBranches.empty())
      return success();
    for (auto br : redundantBranches) {
      handshake::ConditionalBranchOp redunBr =
          cast<handshake::ConditionalBranchOp>(br);
      rewriter.replaceAllUsesWith(redunBr.getTrueResult(),
                                  condBranchOp.getTrueResult());
      rewriter.replaceAllUsesWith(redunBr.getFalseResult(),
                                  condBranchOp.getFalseResult());
      rewriter.eraseOp(br);
    }
    llvm::dbgs() << "\t***Combine Branches Same Signs***\n";
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
    patterns.add<CombineBranchesSameSign, CombineBranchesOppositeSign,
                 CombineInits, CombineMuxes, RemoveDoubleSinkBranches>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::ftd::combineSteeringLogic() {
  return std::make_unique<HandshakeCombineSteeringLogicPass>();
}
