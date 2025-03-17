//===-HandshakeCombineSteeringLogic.cpp - Combines multiple Branches (and
// Merges as well as Muxes) that are having the same input but feeding different
// outputs
//----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeCombineSteeringLogic.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <iterator>
#include <vector>

using namespace mlir;
using namespace dynamatic;

namespace {

  struct CombineInits : public OpRewritePattern<handshake::MergeOp> {
    using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
      PatternRewriter& rewriter) const override {
      // Doublecheck that the Merge has 2 inputs
      if (mergeOp->getNumOperands() != 2)
        return failure();

      // One of the inputs of the Merge must be a constant; otherwise the pattern
      // match fails
      if (!isa_and_nonnull<handshake::ConstantOp>(
        mergeOp.getDataOperands()[0].getDefiningOp()) &&
        !isa_and_nonnull<handshake::ConstantOp>(
          mergeOp.getDataOperands()[1].getDefiningOp()))
        return failure();

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

      for (auto user : mergeOp.getDataOperands()[loopCondOperandIdx].getUsers())
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

      //llvm::errs() << "\t***Combine INITs***\n";
      return success();
    }
  };

  bool isSelfRegenerateMux(handshake::MuxOp muxOp, int& muxOuterInputIdx,
    int& muxCycleInputIdx) {
    // One user must be a Branch; otherwise, the pattern match fails
    bool foundCondBranch = false;
    DenseSet<handshake::ConditionalBranchOp> branches;
    for (auto muxUser : muxOp.getResult().getUsers()) {
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
      PatternRewriter& rewriter) const override {

      // Doublecheck that the Mux has 3 inputs
      if (muxOp.getNumOperands() != 3)
        return failure();

      int muxOuterInputIdx = -1;
      int muxCycleInputIdx = -1;
      if (!isSelfRegenerateMux(muxOp, muxOuterInputIdx, muxCycleInputIdx))
        return failure();

      DenseSet<handshake::MuxOp> conditionMuxUsers;
      DenseSet<handshake::MuxOp> dataMuxUsers;
      DenseSet<handshake::MuxOp> redundantMuxes;

      // Get users of the operation at the muxOuterInputIdx
      for (auto dataUser : muxOp.getDataOperands()[muxOuterInputIdx].getUsers())
        if (isa_and_nonnull<handshake::MuxOp>(dataUser) && dataUser != muxOp) {
          int tempMuxouterInputIdx = -1;
          int tempMuxCycleInputIdx = -1;
          if (isSelfRegenerateMux(cast<handshake::MuxOp>(dataUser),
            tempMuxouterInputIdx, tempMuxCycleInputIdx))
            dataMuxUsers.insert(cast<handshake::MuxOp>(dataUser));
        }

      // Get users of the operation at the select (condition) of the Mux
      for (auto selUser : muxOp.getSelectOperand().getUsers())
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
        return failure();

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

      //llvm::errs() << "\t***Combine Muxes***\n";
      return success();
    }
  };

  /// Remove Conditional Branches that have no successors
  struct RemoveDoubleSinkBranches
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
    using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
      PatternRewriter& rewriter) const override {
      Value branchTrueResult = condBranchOp.getTrueResult();
      Value branchFalseResult = condBranchOp.getFalseResult();

      // Pattern match fails if the Branch has a true or false successor
      if (!branchTrueResult.getUsers().empty() ||
        !branchFalseResult.getUsers().empty())
        return failure();

      rewriter.eraseOp(condBranchOp);

      return success();
    }
  };

  struct CombineBranchesOppositeSign
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
    using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
      PatternRewriter& rewriter) const override {

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
        for (auto condUser : actualCondOperand.getUsers())
          if (isa_and_nonnull<handshake::ConditionalBranchOp>(condUser) &&
            condUser != condBranchOp)
            conditionBranchUsers.insert(
              cast<handshake::ConditionalBranchOp>(condUser));
      }
      else {
        // Do not directly store all users of the condition; rather, store the
        // Branch users of any NOT that is itself a user of the condition
        for (auto condUser : actualCondOperand.getUsers())
          if (isa_and_nonnull<handshake::NotOp>(condUser)) {
            handshake::NotOp notOp = cast<handshake::NotOp>(condUser);
            for (auto notOpUser : notOp.getResult().getUsers()) {
              if (isa_and_nonnull<handshake::ConditionalBranchOp>(notOpUser))
                conditionBranchUsers.insert(
                  cast<handshake::ConditionalBranchOp>(notOpUser));
            }
          }
      }

      for (auto dataUser : dataOperand.getUsers())
        if (isa_and_nonnull<handshake::ConditionalBranchOp>(dataUser) &&
          dataUser != condBranchOp)
          dataBranchUsers.insert(cast<handshake::ConditionalBranchOp>(dataUser));

      if (conditionBranchUsers.empty() || dataBranchUsers.empty())
        return failure();

      // Loop over dataBranchUsers and consider those that are found in
      // conditionBranchUsers
      for (auto br : dataBranchUsers)
        if (conditionBranchUsers.find(br) != conditionBranchUsers.end())
          redundantBranches.insert(br);

      if (redundantBranches.empty())
        return failure();

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

      //llvm::errs() << "\t***Combine Branches Oppostie Signs***\n";
      return success();
    }
  };

  struct CombineBranchesSameSign
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
    using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
      PatternRewriter& rewriter) const override {
      // 1st step: Get the input data value and input condition value
      // 2nd step: Get the users of both the data value and condition value, and
      // if there are common users that are of type Branch, they should be all
      // combined to a single Branch
      Value dataOperand = condBranchOp.getDataOperand();
      Value condOperand = condBranchOp.getConditionOperand();
      DenseSet<handshake::ConditionalBranchOp> conditionBranchUsers;
      DenseSet<handshake::ConditionalBranchOp> dataBranchUsers;
      DenseSet<handshake::ConditionalBranchOp> redundantBranches;

      for (auto condUser : condOperand.getUsers())
        if (isa_and_nonnull<handshake::ConditionalBranchOp>(condUser) &&
          condUser != condBranchOp)
          conditionBranchUsers.insert(
            cast<handshake::ConditionalBranchOp>(condUser));

      for (auto dataUser : dataOperand.getUsers())
        if (isa_and_nonnull<handshake::ConditionalBranchOp>(dataUser) &&
          dataUser != condBranchOp)
          dataBranchUsers.insert(cast<handshake::ConditionalBranchOp>(dataUser));

      if (conditionBranchUsers.empty() || dataBranchUsers.empty())
        return failure();

      // Loop over dataBranchUsers and consider those that are found in
      // conditionBranchUsers
      for (auto br : dataBranchUsers)
        if (conditionBranchUsers.find(br) != conditionBranchUsers.end())
          redundantBranches.insert(br);

      if (redundantBranches.empty())
        return failure();

      for (auto br : redundantBranches) {
        handshake::ConditionalBranchOp redunBr =
          cast<handshake::ConditionalBranchOp>(br);
        rewriter.replaceAllUsesWith(redunBr.getTrueResult(),
          condBranchOp.getTrueResult());
        rewriter.replaceAllUsesWith(redunBr.getFalseResult(),
          condBranchOp.getFalseResult());
        rewriter.eraseOp(br);
      }

      //llvm::errs() << "\t***Combine Branches Same Signs***\n";
      return success();
    }
  };

  struct ConvertLoopMergeToMux : public OpRewritePattern<handshake::MergeOp> {
    using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
      PatternRewriter& rewriter) const override {
      // Doublecheck that the Merge has 2 inputs
      if (mergeOp->getNumOperands() != 2)
        return failure();

      // Get the users of the Merge
      auto mergeUsers = (mergeOp.getResult()).getUsers();
      if (mergeUsers.empty())
        return failure();

      // One user must be a Branch; otherwise, the pattern match fails
      bool foundCondBranch = false;
      DenseSet<handshake::ConditionalBranchOp> branches;
      for (auto mergeUser : mergeUsers) {
        if (isa_and_nonnull<handshake::ConditionalBranchOp>(mergeUser)) {
          foundCondBranch = true;
          branches.insert(cast<handshake::ConditionalBranchOp>(mergeUser));
        }
      }
      if (!foundCondBranch)
        return failure();

      // This condBranchOp must also be an operand forming a cycle with the
      // merge; otherwise, the pattern match fails
      bool foundCycle = false;
      int operIdx = 0;
      int mergeOuterInputIdx = 0;
      int mergeCycleInputIdx = 0;
      handshake::ConditionalBranchOp condBranchOp;
      for (auto mergeOperand : mergeOp->getOperands()) {
        if (isa_and_nonnull<handshake::ConditionalBranchOp>(
          mergeOperand.getDefiningOp()))
          if (branches.contains(cast<handshake::ConditionalBranchOp>(
            mergeOperand.getDefiningOp()))) {
            foundCycle = true;
            mergeCycleInputIdx = operIdx;
            condBranchOp = cast<handshake::ConditionalBranchOp>(
              mergeOperand.getDefiningOp());
            break;
          }
        operIdx++;
      }
      if (!foundCycle)
        return failure();

      // if (!OPTIM_BRANCH_TO_SUPP) {
      //  New condition: The condBranchOp has to be a suppress; otherwise,
      //  the pattern match fails
      if (!condBranchOp.getTrueResult().getUsers().empty() ||
        condBranchOp.getFalseResult().getUsers().empty())
        return failure();
      //}

      mergeOuterInputIdx = (mergeCycleInputIdx == 0) ? 1 : 0;

      // Retrieve the values at the merge inputs
      OperandRange mergeDataOperands = mergeOp.getDataOperands();
      Value mergeInnerOperand = mergeDataOperands[mergeCycleInputIdx];

      // Identify the output of the Branch going outside of the loop (even if it
      // has no users)
      bool isTrueOutputOuter = false;
      Value branchTrueResult = condBranchOp.getTrueResult();
      Value branchFalseResult = condBranchOp.getFalseResult();
      Value branchOuterResult;
      if (branchTrueResult == mergeInnerOperand)
        branchOuterResult = branchFalseResult;
      else if (branchFalseResult == mergeInnerOperand) {
        branchOuterResult = branchTrueResult;
        isTrueOutputOuter = true;
      }
      else
        return failure();

      // 1st) Identify whether the loop condition will be connected directly or
      // through a NOT
      // Note: This strategy is correct, but might result in the insertion of
      // double NOT
      Value condition = condBranchOp.getConditionOperand();
      bool needNot = ((isTrueOutputOuter && mergeCycleInputIdx == 1) ||
        (!isTrueOutputOuter && mergeCycleInputIdx == 0));
      Value iterCond;
      if (needNot) {

        // Check if the condition already feeds a NOT, no need to create a new one
        bool foundNot = false;
        handshake::NotOp existingNotOp;
        for (auto condRes : condition.getUsers()) {
          if (isa_and_nonnull<handshake::NotOp>(condRes)) {
            foundNot = true;
            existingNotOp = cast<handshake::NotOp>(condRes);
            break;
          }
        }
        if (foundNot) {
          iterCond = existingNotOp.getResult();
        }
        else {
          rewriter.setInsertionPoint(condBranchOp);
          handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
            condBranchOp->getLoc(), condition);
          inheritBB(condBranchOp, notOp);
          iterCond = notOp.getResult();
        }

      }
      else {
        iterCond = condition;
      }

      // 2nd) Identify the value of the constant that will be triggered from Start
      // and add it
      // The value of the constant should be the mergeOuterInputIdx
      int constVal = mergeOuterInputIdx;
      // Obtain the start signal from the last argument of any block
      Block* mergeBlock = mergeOp->getBlock();
      MutableArrayRef<BlockArgument> l = mergeBlock->getArguments();
      if (l.empty())
        return failure();
      mlir::Value start = l.back();
      if (!isa<NoneType>(start.getType()))
        return failure();

      // Check if there is an already existing INIT, i.e., a Merge fed from the
      // iterCond
      bool foundInit = false;
      handshake::MergeOp existingInit;
      for (auto iterCondRes : iterCond.getUsers()) {
        if (isa_and_nonnull<handshake::MergeOp>(iterCondRes)) {
          foundInit = true;
          existingInit = cast<handshake::MergeOp>(iterCondRes);
          break;
        }
      }
      Value muxSel;
      if (foundInit) {
        muxSel = existingInit.getResult();
      }
      else {
        // Create a new ConstantOp in the same block as that of the branch
        // forming the cycle
        Type constantType = rewriter.getIntegerType(1);
        rewriter.setInsertionPoint(mergeOp);
        Value valueOfConstant = rewriter.create<handshake::ConstantOp>(
          mergeOp->getLoc(), constantType,
          rewriter.getIntegerAttr(constantType, constVal), start);

        // 3rd) Add a new Merge operation to serve as the INIT
        ValueRange operands = { iterCond, valueOfConstant };
        rewriter.setInsertionPoint(mergeOp);
        handshake::MergeOp initMergeOp =
          rewriter.create<handshake::MergeOp>(mergeOp.getLoc(), operands);
        inheritBB(mergeOp, initMergeOp);

        muxSel = initMergeOp.getResult();
      }

      // Create a new muxOp and make it replace the mergeOp
      rewriter.setInsertionPoint(mergeOp);
      handshake::MuxOp newMuxOp = rewriter.create<handshake::MuxOp>(
        mergeOp.getLoc(), muxSel, mergeOp->getOperands());
      rewriter.replaceOp(mergeOp, newMuxOp);
      inheritBB(mergeOp, newMuxOp);

      // llvm::errs() << "\t***Converted Merge to Mux!***\n";

      return success();
    }
  };

  /// Simple driver for the Handshake Combine Branches Merges pass, based on a
  /// greedy pattern rewriter.
  struct HandshakeCombineSteeringLogicPass
    : public dynamatic::impl::HandshakeCombineSteeringLogicBase<
    HandshakeCombineSteeringLogicPass> {

    void runDynamaticPass() override {
      MLIRContext* ctx = &getContext();
      ModuleOp mod = getOperation();

      GreedyRewriteConfig config;
      config.useTopDownTraversal = true;
      config.enableRegionSimplification = false;
      RewritePatternSet patterns(ctx);
      patterns
        .add<CombineBranchesSameSign, CombineBranchesOppositeSign, CombineInits,
        CombineMuxes, RemoveDoubleSinkBranches, ConvertLoopMergeToMux>(
          ctx);

      if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
        return signalPassFailure();
    };
  };
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::combineSteeringLogic() {
  return std::make_unique<HandshakeCombineSteeringLogicPass>();
}