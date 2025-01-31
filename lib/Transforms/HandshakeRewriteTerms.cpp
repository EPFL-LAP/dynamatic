//===-HandshakeRewriteTerms.cpp - Rewrite Terms in Handshake Operation Sequences
//----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements rewrite patterns for the Handshake rewrite terms pass, which are
// greedily applied on the IR. The pass looks for certain sequences of handshake
// operations and simplifies them. The pass preserves the behaviour of the
// circuit.
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeRewriteTerms.h"
#include "dynamatic/Dialect/Handshake/HandshakeCanonicalize.h"
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

#define OPTIM_DISTR true
#define OPTIM_BRANCH_TO_SUPP true

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
      return failure();

    rewriter.eraseOp(condBranchOp);

    return success();
  }
};

/// Remove floating cycles
struct RemoveMergeFloatingLoop : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
    // Doublecheck that the Merge has 2 inputs
    if (mergeOp->getNumOperands() != 2)
      return failure();

    // Get the users of the Mux
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

    // One of the branches in the set of Branches must be also be an operand
    // forming a cycle with the merge; otherwise, the pattern match fails
    bool foundCycle = false;
    int operIdx = 0;
    int mergeOuterInputIdx = 0;
    int mergeCycleInputIdx = 0;
    handshake::ConditionalBranchOp iterCondBranchOp;
    for (auto mergeOperand : mergeOp.getDataOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              mergeOperand.getDefiningOp()))
        if (branches.contains(cast<handshake::ConditionalBranchOp>(
                mergeOperand.getDefiningOp()))) {
          foundCycle = true;
          mergeCycleInputIdx = operIdx;
          iterCondBranchOp = cast<handshake::ConditionalBranchOp>(
              mergeOperand.getDefiningOp());
          break;
        }
      operIdx++;
    }
    if (!foundCycle)
      return failure();
    mergeOuterInputIdx = (mergeCycleInputIdx == 0) ? 1 : 0;

    // We rely on the RemoveUselessBranches to remove exitCondBranchOp if the
    // only user of the exitCondBranchOp is the Mux
    if (std::distance(mergeUsers.begin(), mergeUsers.end()) != 1 ||
        (std::distance(iterCondBranchOp.getTrueResult().getUsers().begin(),
                       iterCondBranchOp.getTrueResult().getUsers().end()) +
             std::distance(
                 iterCondBranchOp.getFalseResult().getUsers().begin(),
                 iterCondBranchOp.getFalseResult().getUsers().end()) !=
         1))
      return failure();

    rewriter.replaceAllUsesWith(iterCondBranchOp.getDataOperand(),
                                mergeOp.getDataOperands()[mergeOuterInputIdx]);
    rewriter.eraseOp(mergeOp);
    rewriter.eraseOp(iterCondBranchOp);

    return success();
  }
};

/// Remove floating cycles
struct RemoveMuxFloatingLoop : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    // Doublecheck that the Mux has 3 inputs
    if (muxOp->getNumOperands() != 3)
      return failure();

    // Get the users of the Mux
    auto muxUsers = (muxOp.getResult()).getUsers();
    if (muxUsers.empty())
      return failure();

    // One user must be a Branch; otherwise, the pattern match fails
    bool foundCondBranch = false;
    DenseSet<handshake::ConditionalBranchOp> branches;
    for (auto muxUser : muxUsers) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(muxUser)) {
        foundCondBranch = true;
        branches.insert(cast<handshake::ConditionalBranchOp>(muxUser));
      }
    }
    if (!foundCondBranch)
      return failure();

    // One of the branches in the set of Branches must be also be an operand
    // forming a cycle with the merge; otherwise, the pattern match fails
    bool foundCycle = false;
    int operIdx = 0;
    int muxOuterInputIdx = 0;
    int muxCycleInputIdx = 0;
    handshake::ConditionalBranchOp iterCondBranchOp;
    for (auto muxOperand : muxOp.getDataOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              muxOperand.getDefiningOp()))
        if (branches.contains(cast<handshake::ConditionalBranchOp>(
                muxOperand.getDefiningOp()))) {
          foundCycle = true;
          muxCycleInputIdx = operIdx;
          iterCondBranchOp =
              cast<handshake::ConditionalBranchOp>(muxOperand.getDefiningOp());
          break;
        }
      operIdx++;
    }
    if (!foundCycle)
      return failure();
    muxOuterInputIdx = (muxCycleInputIdx == 0) ? 1 : 0;

    // We rely on the RemoveUselessBranches to remove exitCondBranchOp if the
    // only user of the exitCondBranchOp is the Mux
    if (std::distance(muxUsers.begin(), muxUsers.end()) != 1 ||
        (std::distance(iterCondBranchOp.getTrueResult().getUsers().begin(),
                       iterCondBranchOp.getTrueResult().getUsers().end()) +
             std::distance(
                 iterCondBranchOp.getFalseResult().getUsers().begin(),
                 iterCondBranchOp.getFalseResult().getUsers().end()) !=
         1))
      return failure();

    rewriter.replaceAllUsesWith(iterCondBranchOp.getDataOperand(),
                                muxOp.getDataOperands()[muxOuterInputIdx]);
    rewriter.eraseOp(muxOp);
    rewriter.eraseOp(iterCondBranchOp);

    return success();
  }
};

// Removes Conditional Branch and Merge
// operation pairs if both the inputs of the Merge are outputs of the
// Conditional Branch. The results of the Merge are replaced with the data
// operand of the Conditional Branch.
struct RemoveBranchMergeIfThenElse
    : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {

    if (mergeOp->getNumOperands() != 2)
      return failure();

    // The two operands of the merge should be conditional Branches; otherwise,
    // the pattern match fails
    Operation *firstOperand = mergeOp.getOperands()[0].getDefiningOp();
    Operation *secondOperand = mergeOp.getOperands()[1].getDefiningOp();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(firstOperand) ||
        !isa_and_nonnull<handshake::ConditionalBranchOp>(secondOperand))
      return failure();

    handshake::ConditionalBranchOp firstBranchOperand =
        cast<handshake::ConditionalBranchOp>(firstOperand);
    handshake::ConditionalBranchOp secondBranchOperand =
        cast<handshake::ConditionalBranchOp>(secondOperand);

    if (!OPTIM_BRANCH_TO_SUPP) {
      // New conditions: to ensure we only conside suppresses
      // If the first branch is not a suppress, the pattern match fails
      if ((!firstBranchOperand.getTrueResult().getUsers().empty()) ||
          (firstBranchOperand.getTrueResult().getUsers().empty() &&
           firstBranchOperand.getFalseResult().getUsers().empty()))
        return failure();
      // If the second branch is not a suppress, the pattern match fails
      if ((!secondBranchOperand.getTrueResult().getUsers().empty()) ||
          (secondBranchOperand.getTrueResult().getUsers().empty() &&
           secondBranchOperand.getFalseResult().getUsers().empty()))
        return failure();
    }

    if (!OPTIM_DISTR) {
      if (std::distance(firstBranchOperand.getFalseResult().getUsers().begin(),
                        firstBranchOperand.getFalseResult().getUsers().end()) !=
          1)
        return failure();
      if (std::distance(
              secondBranchOperand.getFalseResult().getUsers().begin(),
              secondBranchOperand.getFalseResult().getUsers().end()) != 1)
        return failure();
    }
    Value firstBranchCondition = firstBranchOperand.getConditionOperand();
    Value firstOriginalBranchCondition = firstBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(firstBranchCondition.getDefiningOp()))
      firstOriginalBranchCondition =
          firstBranchCondition.getDefiningOp()->getOperand(0);

    Value secondBranchCondition = secondBranchOperand.getConditionOperand();
    Value secondOriginalBranchCondition = secondBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(
            secondBranchCondition.getDefiningOp()))
      secondOriginalBranchCondition =
          secondBranchCondition.getDefiningOp()->getOperand(0);

    // If the two original conditions are not equivalent, the pattern match
    // fails
    if (firstOriginalBranchCondition != secondOriginalBranchCondition)
      return failure();

    // If the data input of the two Branches is not the same, the pattern match
    // fails
    Value firstBranchData = firstBranchOperand.getDataOperand();
    Value secondBranchData = secondBranchOperand.getDataOperand();
    if (firstBranchData != secondBranchData)
      return failure();

    Value mergeOutput = mergeOp.getResult();

    // Replace all uses of the merge output with the input of the Branches
    rewriter.replaceAllUsesWith(mergeOutput, firstBranchData);
    // Delete the merge
    rewriter.eraseOp(mergeOp);
    return success();
  }
};

// Removes Conditional Branch and Mux
// operation pairs if both the inputs of the Mux are outputs of the Conditional
// Branch. The results of the MeMuxrge are replaced with the data operand.
struct RemoveBranchMuxIfThenElse : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {

    if (muxOp->getNumOperands() != 3)
      return failure();

    // The two operands of the mux should be conditional Branches; otherwise,
    // the pattern match fails
    Operation *firstOperand = muxOp.getDataOperands()[0].getDefiningOp();
    Operation *secondOperand = muxOp.getDataOperands()[1].getDefiningOp();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(firstOperand) ||
        !isa_and_nonnull<handshake::ConditionalBranchOp>(secondOperand))
      return failure();

    handshake::ConditionalBranchOp firstBranchOperand =
        cast<handshake::ConditionalBranchOp>(firstOperand);
    handshake::ConditionalBranchOp secondBranchOperand =
        cast<handshake::ConditionalBranchOp>(secondOperand);

    if (!OPTIM_BRANCH_TO_SUPP) {
      if ((!firstBranchOperand.getTrueResult().getUsers().empty()) ||
          (firstBranchOperand.getTrueResult().getUsers().empty() &&
           firstBranchOperand.getFalseResult().getUsers().empty()))
        return failure();
      // If the second branch is not a suppress, the pattern match fails
      if ((!secondBranchOperand.getTrueResult().getUsers().empty()) ||
          (secondBranchOperand.getTrueResult().getUsers().empty() &&
           secondBranchOperand.getFalseResult().getUsers().empty()))
        return failure();
    }

    if (!OPTIM_DISTR) {
      if (std::distance(firstBranchOperand.getFalseResult().getUsers().begin(),
                        firstBranchOperand.getFalseResult().getUsers().end()) !=
          1)
        return failure();
      if (std::distance(
              secondBranchOperand.getFalseResult().getUsers().begin(),
              secondBranchOperand.getFalseResult().getUsers().end()) != 1)
        return failure();
    }

    Value firstBranchCondition = firstBranchOperand.getConditionOperand();
    Value firstOriginalBranchCondition = firstBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(firstBranchCondition.getDefiningOp()))
      firstOriginalBranchCondition =
          firstBranchCondition.getDefiningOp()->getOperand(0);

    Value secondBranchCondition = secondBranchOperand.getConditionOperand();
    Value secondOriginalBranchCondition = secondBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(
            secondBranchCondition.getDefiningOp()))
      secondOriginalBranchCondition =
          secondBranchCondition.getDefiningOp()->getOperand(0);

    // If the two original conditions are not equivalent, the pattern match
    // fails
    if (firstOriginalBranchCondition != secondOriginalBranchCondition)
      return failure();

    // If the data input of the two Branches is not the same, the pattern match
    // fails
    Value firstBranchData = firstBranchOperand.getDataOperand();
    Value secondBranchData = secondBranchOperand.getDataOperand();
    if (firstBranchData != secondBranchData)
      return failure();

    Value muxOutput = muxOp.getResult();

    // Replace all uses of the mux output with the input of the Branches
    rewriter.replaceAllUsesWith(muxOutput, firstBranchData);
    // Delete the mux
    rewriter.eraseOp(muxOp);

    return success();
  }
};

// Removes Merge and Branch operation pairs there exits a loop between
// the Merge and the Branch.
struct RemoveMergeBranchLoop : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
    // Doublecheck that the Merge has 2 inputs
    if (mergeOp->getNumOperands() != 2)
      return failure();

    // Get the users of the Merge
    auto mergeUsers = (mergeOp.getResult()).getUsers();

    if (!OPTIM_DISTR) {
      if (std::distance(mergeUsers.begin(), mergeUsers.end()) != 2)
        return failure();
    }

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

    // One of the branches in the set of Branches must be also be an operand
    // forming a cycle with the merge; otherwise, the pattern match fails
    bool foundCycle = false;
    int operIdx = 0;
    int mergeOuterInputIdx = 0;
    int mergeCycleInputIdx = 0;
    handshake::ConditionalBranchOp iterCondBranchOp;
    for (auto mergeOperand : mergeOp.getDataOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              mergeOperand.getDefiningOp()))
        if (branches.contains(cast<handshake::ConditionalBranchOp>(
                mergeOperand.getDefiningOp()))) {
          foundCycle = true;
          mergeCycleInputIdx = operIdx;
          iterCondBranchOp = cast<handshake::ConditionalBranchOp>(
              mergeOperand.getDefiningOp());
          break;
        }
      operIdx++;
    }
    if (!foundCycle)
      return failure();

    mergeOuterInputIdx = (mergeCycleInputIdx == 0) ? 1 : 0;

    // One of the branches in the set of Branches must have the same condition
    // as that of iterCondBranchOp but has none of its succs equal to the merge
    Value iterBranchCondition = iterCondBranchOp.getConditionOperand();
    Value iterOriginalBranchCondition = iterBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(iterBranchCondition.getDefiningOp()))
      iterOriginalBranchCondition =
          iterBranchCondition.getDefiningOp()->getOperand(0);

    // This will be used to check inside the loop that the condition of the exit
    // branch has the same value but opposite sign
    bool firstPassWithTrueCond =
        ((iterCondBranchOp.getTrueResult() ==
              mergeOp.getDataOperands()[mergeCycleInputIdx] &&
          iterBranchCondition == iterOriginalBranchCondition) ||
         (iterCondBranchOp.getFalseResult() ==
              mergeOp.getDataOperands()[mergeCycleInputIdx] &&
          iterBranchCondition != iterOriginalBranchCondition));

    bool foundExitBranch = false;
    handshake::ConditionalBranchOp exitCondBranchOp;
    Value exitBranchCondition;
    Value exitOriginalBranchCondition;

    for (auto br : branches) {
      // Identify the brBranch condition to compare it to that of the iterBranch
      Value brBranchCondition = br.getConditionOperand();
      Value brOriginalBranchCondition = brBranchCondition;
      if (isa_and_nonnull<handshake::NotOp>(brBranchCondition.getDefiningOp()))
        brOriginalBranchCondition =
            brBranchCondition.getDefiningOp()->getOperand(0);

      Value brTrueResult = br.getTrueResult();
      Value brFalseResult = br.getFalseResult();
      // Make sure that this Branch is having at least 1 succ that is not making
      // a cycle with the Merge
      if (brTrueResult != mergeOp.getDataOperands()[mergeCycleInputIdx] ||
          brFalseResult != mergeOp.getDataOperands()[mergeCycleInputIdx]) {

        if (br == iterCondBranchOp && !br.getTrueResult().getUsers().empty() &&
            !br.getFalseResult().getUsers().empty()) {
          foundExitBranch = true;
          exitCondBranchOp = br;
          exitBranchCondition = brBranchCondition;
          exitOriginalBranchCondition = brOriginalBranchCondition;
          break;
        } else {
          bool passWithTrueCond =
              ((!br.getTrueResult().getUsers().empty() &&
                brBranchCondition == brOriginalBranchCondition) ||
               (!br.getFalseResult().getUsers().empty() &&
                brBranchCondition != brOriginalBranchCondition));

          bool consistentSigns = (passWithTrueCond == firstPassWithTrueCond);
          if (brOriginalBranchCondition == iterOriginalBranchCondition &&
              !consistentSigns) {
            foundExitBranch = true;
            exitCondBranchOp = br;
            exitBranchCondition = brBranchCondition;
            exitOriginalBranchCondition = brOriginalBranchCondition;
            break;
          }
        }
      }
    }

    if (!foundExitBranch)
      return failure();

    // Pattern match fails if both outputs of the exitCondBranchOp are empty
    if (exitCondBranchOp.getTrueResult().getUsers().empty() &&
        exitCondBranchOp.getFalseResult().getUsers().empty())
      return failure();

    if (!OPTIM_BRANCH_TO_SUPP) {
      if ((!exitCondBranchOp.getTrueResult().getUsers().empty()) ||
          (exitCondBranchOp.getTrueResult().getUsers().empty() &&
           exitCondBranchOp.getFalseResult().getUsers().empty()))
        return failure();
      if ((!iterCondBranchOp.getTrueResult().getUsers().empty()) ||
          (iterCondBranchOp.getTrueResult().getUsers().empty() &&
           iterCondBranchOp.getFalseResult().getUsers().empty()))
        return failure();
    }

    // Doublecheck which output of the exitCondBranchOp is not empty and is not
    // feeding the cycle of the Merge
    Value brTrueResult = exitCondBranchOp.getTrueResult();
    Value brFalseResult = exitCondBranchOp.getFalseResult();
    if (brTrueResult != mergeOp.getDataOperands()[mergeCycleInputIdx] &&
        !brTrueResult.getUsers().empty()) {
      rewriter.replaceAllUsesWith(
          brTrueResult, mergeOp.getDataOperands()[mergeOuterInputIdx]);
    } else {
      if (brFalseResult != mergeOp.getDataOperands()[mergeCycleInputIdx] &&
          !brFalseResult.getUsers().empty()) {
        rewriter.replaceAllUsesWith(
            brFalseResult, mergeOp.getDataOperands()[mergeOuterInputIdx]);
      }
    }

    return success();
  }
};

// Removes Mux and Branch operation pairs there exits a loop between
// the Mux and the Branch.
struct RemoveMuxBranchLoop : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    // Doublecheck that the Mux has 3 inputs
    if (muxOp->getNumOperands() != 3)
      return failure();

    // Get the users of the Mux
    auto muxUsers = (muxOp.getResult()).getUsers();

    if (!OPTIM_DISTR) {
      if (std::distance(muxUsers.begin(), muxUsers.end()) != 2)
        return failure();
    }

    // One user must be a Branch; otherwise, the pattern match fails
    bool foundCondBranch = false;
    DenseSet<handshake::ConditionalBranchOp> branches;
    for (auto muxUser : muxUsers) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(muxUser)) {
        foundCondBranch = true;
        branches.insert(cast<handshake::ConditionalBranchOp>(muxUser));
      }
    }
    if (!foundCondBranch)
      return failure();

    bool foundCycle = false;
    int operIdx = 0;
    int muxOuterInputIdx = 0;
    int muxCycleInputIdx = 0;
    handshake::ConditionalBranchOp iterCondBranchOp;
    for (auto muxOperand : muxOp.getDataOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              muxOperand.getDefiningOp()))
        if (branches.contains(cast<handshake::ConditionalBranchOp>(
                muxOperand.getDefiningOp()))) {
          foundCycle = true;
          muxCycleInputIdx = operIdx;
          iterCondBranchOp =
              cast<handshake::ConditionalBranchOp>(muxOperand.getDefiningOp());
          break;
        }
      operIdx++;
    }
    if (!foundCycle)
      return failure();

    muxOuterInputIdx = (muxCycleInputIdx == 0) ? 1 : 0;

    // One of the branches in the set of Branches must have the same condition
    // as that of iterCondBranchOp but has none of its succs equal to the Cmerge
    Value iterBranchCondition = iterCondBranchOp.getConditionOperand();
    Value iterOriginalBranchCondition = iterBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(iterBranchCondition.getDefiningOp()))
      iterOriginalBranchCondition =
          iterBranchCondition.getDefiningOp()->getOperand(0);

    // This will be used to check inside the loop that the condition of the exit
    // branch has the same value but opposite sign
    bool firstPassWithTrueCond =
        ((iterCondBranchOp.getTrueResult() ==
              muxOp.getDataOperands()[muxCycleInputIdx] &&
          iterBranchCondition == iterOriginalBranchCondition) ||
         (iterCondBranchOp.getFalseResult() ==
              muxOp.getDataOperands()[muxCycleInputIdx] &&
          iterBranchCondition != iterOriginalBranchCondition));

    bool foundExitBranch = false;
    handshake::ConditionalBranchOp exitCondBranchOp;
    Value exitBranchCondition;
    Value exitOriginalBranchCondition;

    for (auto br : branches) {
      Value brBranchCondition = br.getConditionOperand();
      Value brOriginalBranchCondition = brBranchCondition;
      if (isa_and_nonnull<handshake::NotOp>(brBranchCondition.getDefiningOp()))
        brOriginalBranchCondition =
            brBranchCondition.getDefiningOp()->getOperand(0);

      Value brTrueResult = br.getTrueResult();
      Value brFalseResult = br.getFalseResult();
      // Make sure that this Branch is having at least 1 succ that is not making
      // a cycle with the Merge
      if (brTrueResult != muxOp.getDataOperands()[muxCycleInputIdx] ||
          brFalseResult != muxOp.getDataOperands()[muxCycleInputIdx]) {

        if (br == iterCondBranchOp && !br.getTrueResult().getUsers().empty() &&
            !br.getFalseResult().getUsers().empty()) {
          foundExitBranch = true;
          exitCondBranchOp = br;
          exitBranchCondition = brBranchCondition;
          exitOriginalBranchCondition = brOriginalBranchCondition;
          break;
        } else {
          bool passWithTrueCond =
              ((!br.getTrueResult().getUsers().empty() &&
                brBranchCondition == brOriginalBranchCondition) ||
               (!br.getFalseResult().getUsers().empty() &&
                brBranchCondition != brOriginalBranchCondition));

          bool consistentSigns = (passWithTrueCond == firstPassWithTrueCond);
          if (brOriginalBranchCondition == iterOriginalBranchCondition &&
              !consistentSigns) {
            foundExitBranch = true;
            exitCondBranchOp = br;
            exitBranchCondition = brBranchCondition;
            exitOriginalBranchCondition = brOriginalBranchCondition;
            break;
          }
        }
      }
    }

    if (!foundExitBranch)
      return failure();

    // Pattern match fails if both outputs of the exitCondBranchOp are empty
    if (exitCondBranchOp.getTrueResult().getUsers().empty() &&
        exitCondBranchOp.getFalseResult().getUsers().empty())
      return failure();

    if (!OPTIM_BRANCH_TO_SUPP) {
      // This exitCondBranchOp has to be a suppress with at least one succ in
      // only one side; otherwise, the pattern match fails
      if ((!exitCondBranchOp.getTrueResult().getUsers().empty()) ||
          (exitCondBranchOp.getTrueResult().getUsers().empty() &&
           exitCondBranchOp.getFalseResult().getUsers().empty()))
        return failure();

      // New conditions: the iterCondBranch must also be a suppress; otherwise
      // the pattern match fails If the first branch is not a suppress, the
      // pattern match fails
      if ((!iterCondBranchOp.getTrueResult().getUsers().empty()) ||
          (iterCondBranchOp.getTrueResult().getUsers().empty() &&
           iterCondBranchOp.getFalseResult().getUsers().empty()))
        return failure();
    }

    // Doublecheck which output of the exitCondBranchOp is not empty and is not
    // feeding the cycle of the Merge
    Value brTrueResult = exitCondBranchOp.getTrueResult();
    Value brFalseResult = exitCondBranchOp.getFalseResult();
    if (brTrueResult != muxOp.getDataOperands()[muxCycleInputIdx] &&
        !brTrueResult.getUsers().empty()) {
      rewriter.replaceAllUsesWith(brTrueResult,
                                  muxOp.getDataOperands()[muxOuterInputIdx]);
    } else {
      if (brFalseResult != muxOp.getDataOperands()[muxCycleInputIdx] &&
          !brFalseResult.getUsers().empty()) {
        rewriter.replaceAllUsesWith(brFalseResult,
                                    muxOp.getDataOperands()[muxOuterInputIdx]);
      }
    }

    return success();
  }
};

// Extract the index result of the Control Merge in a loop structure.
struct ExtractLoopCondition
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    // Doublecheck that the CMerge has 2 inputs
    if (cmergeOp->getNumOperands() != 2)
      return failure();

    // Get the users of the CMerge
    auto cmergeUsers = (cmergeOp.getResults()).getUsers();
    if (cmergeUsers.empty())
      return failure();

    // One user must be a Branch; otherwise, the pattern match fails
    bool foundCondBranch = false;
    DenseSet<handshake::ConditionalBranchOp> branches;
    for (auto cmergeUser : cmergeUsers) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(cmergeUser)) {
        foundCondBranch = true;
        branches.insert(cast<handshake::ConditionalBranchOp>(cmergeUser));
      }
    }
    if (!foundCondBranch)
      return failure();

    // This condBranchOp must also be an operand forming a cycle with the
    // cmerge; otherwise, the pattern match fails
    bool foundCycle = false;
    int operIdx = 0;
    int cmergeOuterInputIdx = 0;
    int cmergeCycleInputIdx = 0;
    handshake::ConditionalBranchOp condBranchOp;
    for (auto cmergeOperand : cmergeOp->getOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              cmergeOperand.getDefiningOp()))
        if (branches.contains(cast<handshake::ConditionalBranchOp>(
                cmergeOperand.getDefiningOp()))) {
          foundCycle = true;
          cmergeCycleInputIdx = operIdx;
          condBranchOp = cast<handshake::ConditionalBranchOp>(
              cmergeOperand.getDefiningOp());
          break;
        }
      operIdx++;
    }
    if (!foundCycle)
      return failure();

    if (!OPTIM_BRANCH_TO_SUPP) {
      // New condition: The condBranchOp has to be a suppress; otherwise, the
      // pattern match fails
      if (!condBranchOp.getTrueResult().getUsers().empty() ||
          condBranchOp.getFalseResult().getUsers().empty())
        return failure();
    }

    cmergeOuterInputIdx = (cmergeCycleInputIdx == 0) ? 1 : 0;

    // Retrieve the values at the Cmerge inputs
    OperandRange cmergeDataOperands = cmergeOp.getDataOperands();
    Value cmergeInnerOperand = cmergeDataOperands[cmergeCycleInputIdx];

    // Identify the output of the Branch going outside of the loop (even if it
    // has no users)
    bool isTrueOutputOuter = false;
    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();
    Value branchOuterResult;
    if (branchTrueResult == cmergeInnerOperand)
      branchOuterResult = branchFalseResult;
    else if (branchFalseResult == cmergeInnerOperand) {
      branchOuterResult = branchTrueResult;
      isTrueOutputOuter = true;
    } else
      return failure();

    Value condition = condBranchOp.getConditionOperand();
    bool needNot = ((isTrueOutputOuter && cmergeCycleInputIdx == 1) ||
                    (!isTrueOutputOuter && cmergeCycleInputIdx == 0));
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
      } else {
        rewriter.setInsertionPoint(condBranchOp);
        handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
            condBranchOp->getLoc(), condition);
        inheritBB(notOp, condBranchOp);
        iterCond = notOp.getResult();
      }

    } else {
      iterCond = condition;
    }

    int constVal = cmergeOuterInputIdx;
    // Obtain the start signal from the last argument of any block
    Block *cmergeBlock = cmergeOp->getBlock();
    MutableArrayRef<BlockArgument> l = cmergeBlock->getArguments();
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
    } else {
      // Create a new ConstantOp in the same block as that of the branch
      // forming the cycle
      Type constantType = rewriter.getIntegerType(1);
      Value valueOfConstant = rewriter.create<handshake::ConstantOp>(
          condBranchOp->getLoc(), constantType,
          rewriter.getIntegerAttr(constantType, constVal), start);

      ValueRange operands = {iterCond, valueOfConstant};
      rewriter.setInsertionPoint(cmergeOp);
      handshake::MergeOp mergeOp =
          rewriter.create<handshake::MergeOp>(cmergeOp.getLoc(), operands);
      muxSel = mergeOp.getResult();
      inheritBB(cmergeOp, mergeOp);
    }

    Value index = cmergeOp.getIndex();
    rewriter.replaceAllUsesWith(index, muxSel);

    return success();
  }
};

// Extract the index result of the Control Merge in an if-then-else structure.
struct ExtractIfThenElseCondition
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {

    // Pattern match fails if the cntrlMerge does not have exactly two inputs
    if (cmergeOp->getNumOperands() != 2)
      return failure();

    // The two operands of the Cmerge should be conditional Branches; otherwise,
    // the pattern match fails
    Operation *firstOperand = cmergeOp.getOperands()[0].getDefiningOp();
    Operation *secondOperand = cmergeOp.getOperands()[1].getDefiningOp();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(firstOperand) ||
        !isa_and_nonnull<handshake::ConditionalBranchOp>(secondOperand))
      return failure();

    handshake::ConditionalBranchOp firstBranchOperand =
        cast<handshake::ConditionalBranchOp>(firstOperand);
    handshake::ConditionalBranchOp secondBranchOperand =
        cast<handshake::ConditionalBranchOp>(secondOperand);

    if (!OPTIM_BRANCH_TO_SUPP) {
      if (!firstBranchOperand.getTrueResult().getUsers().empty() ||
          firstBranchOperand.getFalseResult().getUsers().empty())
        return failure();
      if (!secondBranchOperand.getTrueResult().getUsers().empty() ||
          secondBranchOperand.getFalseResult().getUsers().empty())
        return failure();
    }

    Value firstBranchCondition = firstBranchOperand.getConditionOperand();
    Value firstOriginalBranchCondition = firstBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(firstBranchCondition.getDefiningOp()))
      firstOriginalBranchCondition =
          firstBranchCondition.getDefiningOp()->getOperand(0);

    Value secondBranchCondition = secondBranchOperand.getConditionOperand();
    Value secondOriginalBranchCondition = secondBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(
            secondBranchCondition.getDefiningOp()))
      secondOriginalBranchCondition =
          secondBranchCondition.getDefiningOp()->getOperand(0);

    // If the two original conditions are not equivalent, the pattern match
    // fails
    if (firstOriginalBranchCondition != secondOriginalBranchCondition)
      return failure();

    Value index = cmergeOp.getIndex();

    bool reversedFirstInput =
        (firstBranchOperand.getTrueResult() == cmergeOp.getOperands()[0] &&
         firstBranchCondition == firstOriginalBranchCondition) ||
        (firstBranchOperand.getFalseResult() == cmergeOp.getOperands()[0] &&
         firstBranchCondition != firstOriginalBranchCondition);
    bool reversedSecondInput =
        (secondBranchOperand.getFalseResult() == cmergeOp.getOperands()[1] &&
         secondBranchCondition == secondOriginalBranchCondition) ||
        (secondBranchOperand.getTrueResult() == cmergeOp.getOperands()[1] &&
         secondBranchCondition != secondOriginalBranchCondition);

    bool needNot = reversedFirstInput && reversedSecondInput;
    Value cond;
    if (needNot) {

      // Check if the condition already feeds a NOT, no need to create a new one
      bool foundNot = false;
      handshake::NotOp existingNotOp;
      for (auto condRes : firstOriginalBranchCondition.getUsers()) {
        if (isa_and_nonnull<handshake::NotOp>(condRes)) {
          foundNot = true;
          existingNotOp = cast<handshake::NotOp>(condRes);
          break;
        }
      }

      if (foundNot) {
        cond = existingNotOp.getResult();
      } else {
        rewriter.setInsertionPoint(cmergeOp);
        handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
            cmergeOp->getLoc(), firstOriginalBranchCondition);
        inheritBB(cmergeOp, notOp);
        cond = notOp.getResult();
      }

    } else {
      cond = firstOriginalBranchCondition;
    }

    // Replace the Cmerge index output with the branch condition
    rewriter.replaceAllUsesWith(index, cond);

    return success();
  }
};

// Replaces a pair of consecutive Suppress operations with a
// a single suppress operation with a mux at its condition input.
struct ShortenSuppressPairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(handshake::ConditionalBranchOp firstCondBranchOp,
                  PatternRewriter &rewriter) const override {
    // Consider only Branches that either have trueSuccs or falseSuccs but not
    // both
    Value firstTrueResult = firstCondBranchOp.getTrueResult();
    Value firstFalseResult = firstCondBranchOp.getFalseResult();
    bool firstTrueSuccOnlyFlag = (!firstTrueResult.getUsers().empty() &&
                                  firstFalseResult.getUsers().empty());
    bool firstFalseSuccOnlyFlag = (firstTrueResult.getUsers().empty() &&
                                   !firstFalseResult.getUsers().empty());
    if (!firstTrueSuccOnlyFlag && !firstFalseSuccOnlyFlag)
      return failure();

    // There must be only 1 successor; otherwise, we cannot optimize
    if (std::distance(firstTrueResult.getUsers().begin(),
                      firstTrueResult.getUsers().end()) > 1 ||
        std::distance(firstFalseResult.getUsers().begin(),
                      firstFalseResult.getUsers().end()) > 1)
      return failure();

    Operation *succ = nullptr;
    Value succVal;
    if (firstTrueSuccOnlyFlag) {
      succ = *firstTrueResult.getUsers().begin();
      succVal = firstTrueResult;
    } else {
      succ = *firstFalseResult.getUsers().begin();
      succVal = firstFalseResult;
    }

    // This succ must be a conditional branch; otherwise, the pattern match
    // fails
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(succ))
      return failure();

    handshake::ConditionalBranchOp secondCondBranchOp =
        cast<handshake::ConditionalBranchOp>(succ);

    // The pattern match should fail if this Branch has succs both in the true
    // and false sides
    Value secondTrueResult = secondCondBranchOp.getTrueResult();
    Value secondFalseResult = secondCondBranchOp.getFalseResult();
    bool secondTrueSuccOnlyFlag = (!secondTrueResult.getUsers().empty() &&
                                   secondFalseResult.getUsers().empty());
    bool secondFalseSuccOnlyFlag = (secondTrueResult.getUsers().empty() &&
                                    !secondFalseResult.getUsers().empty());
    if (!secondTrueSuccOnlyFlag && !secondFalseSuccOnlyFlag)
      return failure();

    Value condBr1 = firstCondBranchOp.getConditionOperand();
    Value condBr2 = secondCondBranchOp.getConditionOperand();
    if (firstTrueSuccOnlyFlag && secondFalseSuccOnlyFlag) {

      // Check if the condition already feeds a NOT, no need to create a new one
      bool foundNot = false;
      handshake::NotOp existingNotOp;
      for (auto condRes : condBr1.getUsers()) {
        if (isa_and_nonnull<handshake::NotOp>(condRes)) {
          foundNot = true;
          existingNotOp = cast<handshake::NotOp>(condRes);
          break;
        }
      }
      Value newCond;
      if (foundNot) {
        newCond = existingNotOp.getResult();
      } else {
        // Insert a NOT at the condition input of the first Branch
        rewriter.setInsertionPoint(firstCondBranchOp);
        handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
            firstCondBranchOp->getLoc(), condBr1);
        inheritBB(firstCondBranchOp, notOp);

        newCond = notOp.getResult();
      }

      rewriter.replaceAllUsesWith(condBr1, newCond);

      // Replace all uses coming from the true side of the first Branch with
      // the false side of it
      rewriter.replaceAllUsesWith(firstTrueResult, firstFalseResult);
      // Adjust the firstTrueSuccOnlyFlag and firstFalseSuccOnlyFlag
      firstTrueSuccOnlyFlag = false;
      firstFalseSuccOnlyFlag = true;

      // Retrieve the new value of the condition, in case it is not updated
      condBr1 = firstCondBranchOp.getConditionOperand();
    } else {
      if (firstFalseSuccOnlyFlag && secondTrueSuccOnlyFlag) {
        // Check if the condition already feeds a NOT, no need to create a new
        // one
        bool foundNot = false;
        handshake::NotOp existingNotOp;
        for (auto condRes : condBr2.getUsers()) {
          if (isa_and_nonnull<handshake::NotOp>(condRes)) {
            foundNot = true;
            existingNotOp = cast<handshake::NotOp>(condRes);
            break;
          }
        }
        Value newCond;
        if (foundNot) {
          newCond = existingNotOp.getResult();
        } else {
          // Insert a NOT at the condition input of the second Branch
          rewriter.setInsertionPoint(secondCondBranchOp);
          handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
              secondCondBranchOp->getLoc(), condBr2);
          inheritBB(secondCondBranchOp, notOp);

          newCond = notOp.getResult();
        }

        rewriter.replaceAllUsesWith(condBr2, newCond);

        // Replace all uses coming from the true side of the first Branch with
        // the false side of it
        rewriter.replaceAllUsesWith(secondTrueResult, secondFalseResult);
        // Adjust the secondTrueSuccOnlyFlag and firstFalseSuccOnlyFlag
        secondTrueSuccOnlyFlag = false;
        secondFalseSuccOnlyFlag = true;

        // Retrieve the new value of the condition, in case it is not updated
        condBr2 = secondCondBranchOp.getConditionOperand();
      }
    }

    int64_t constantValue;
    if (firstTrueSuccOnlyFlag) {
      assert(secondTrueSuccOnlyFlag);
      // this means suppress when the condition is false
      constantValue = 0;
    } else {
      assert(firstFalseSuccOnlyFlag && secondFalseSuccOnlyFlag);
      // this means suppress when the condition is true
      constantValue = 1;
    }
    rewriter.setInsertionPoint(secondCondBranchOp);
    Value source =
        rewriter.create<handshake::SourceOp>(secondCondBranchOp->getLoc());

    Type constantType = rewriter.getIntegerType(1);
    Value constantVal = rewriter.create<handshake::ConstantOp>(
        secondCondBranchOp->getLoc(), constantType,
        rewriter.getIntegerAttr(constantType, constantValue), source);

    // Create a new Mux and assign its operands
    ValueRange muxOperands;
    if (firstTrueSuccOnlyFlag) {
      assert(secondTrueSuccOnlyFlag);
      muxOperands = {constantVal, condBr2};
    } else {
      assert(firstFalseSuccOnlyFlag && secondFalseSuccOnlyFlag);
      muxOperands = {condBr2, constantVal};
    }
    rewriter.setInsertionPoint(secondCondBranchOp);
    handshake::MuxOp mux = rewriter.create<handshake::MuxOp>(
        secondCondBranchOp->getLoc(), condBr1, muxOperands);
    inheritBB(secondCondBranchOp, mux);

    // Correct the inputs of the second Branch
    Value muxResult = mux.getResult();
    Value dataOperand = firstCondBranchOp.getDataOperand();
    ValueRange branchOperands = {muxResult, dataOperand};
    secondCondBranchOp->setOperands(branchOperands);

    // Erase the first Branch
    rewriter.eraseOp(firstCondBranchOp);

    return success();
  }
};

// Replaces a pair of consecutive Repeats with a
// a single Repeat with a mux at its condition input.
struct ShortenMuxRepeatPairs : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp firstMuxOp,
                                PatternRewriter &rewriter) const override {
    auto firstMuxUsers = (firstMuxOp.getResult()).getUsers();
    if (std::distance(firstMuxUsers.begin(), firstMuxUsers.end()) != 2)
      return failure();

    // If the mux is not driven by a Merge (i.e., INIT), the pattern match
    // fails
    if (!isa_and_nonnull<handshake::MergeOp>(
            firstMuxOp.getSelectOperand().getDefiningOp()))
      return failure();

    bool firstFoundCondBranch = false;
    handshake::ConditionalBranchOp firstCondBranchOp;
    bool foundSecondMux = false;
    handshake::MuxOp secondMuxOp;
    for (auto muxUser : firstMuxUsers) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(muxUser)) {
        firstFoundCondBranch = true;
        firstCondBranchOp = cast<handshake::ConditionalBranchOp>(muxUser);
      } else if (isa_and_nonnull<handshake::MuxOp>(muxUser)) {
        foundSecondMux = true;
        secondMuxOp = cast<handshake::MuxOp>(muxUser);
      }
    }
    if (!firstFoundCondBranch && !foundSecondMux)
      return failure();

    // The firstCondBranchOp must be also be an operand
    // forming a cycle with the firstMuxOp; otherwise, the pattern match fails
    bool firstFoundCycle = false;
    int operIdx = 0;
    int firstMuxCycleInputIdx = 0;
    for (auto muxOperand : firstMuxOp->getOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              muxOperand.getDefiningOp()))
        if (cast<handshake::ConditionalBranchOp>(muxOperand.getDefiningOp()) ==
            firstCondBranchOp) {
          firstFoundCycle = true;
          firstMuxCycleInputIdx = operIdx;
          break;
        }
      operIdx++;
    }
    if (!firstFoundCycle)
      return failure();
    int firstMuxOuterInputIdx = (firstMuxCycleInputIdx == 0) ? 1 : 0;

    // The firstCondBranchOp should not have any more successors; otherwise,
    // it is not a Repeat structure
    if (std::distance(firstCondBranchOp->getResults().getUsers().begin(),
                      firstCondBranchOp->getResults().getUsers().end()) != 1)
      return failure();

    auto secondMuxUsers = (secondMuxOp.getResult()).getUsers();
    if (secondMuxUsers.empty())
      return failure();

    // If the mux is not driven by a Merge (i.e., INIT), the pattern match
    // fails
    if (!isa_and_nonnull<handshake::MergeOp>(
            secondMuxOp.getSelectOperand().getDefiningOp()))
      return failure();

    // One user must be a Branch; otherwise, the pattern match fails
    bool secondFoundCondBranch = false;
    // This second Repeat could be feeding many users including maybe another
    // non-loop Branch
    DenseSet<handshake::ConditionalBranchOp> branches;
    for (auto muxUser : secondMuxUsers) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(muxUser)) {
        secondFoundCondBranch = true;
        branches.insert(cast<handshake::ConditionalBranchOp>(muxUser));
      }
    }
    if (!secondFoundCondBranch)
      return failure();

    // One of the branches in the set of Branches must be also be an operand
    // forming a cycle with the mux; otherwise, the pattern match fails
    bool secondFoundCycle = false;
    operIdx = 0;
    int secondMuxCycleInputIdx = 0;
    handshake::ConditionalBranchOp secondCondBranchOp;
    for (auto muxOperand : secondMuxOp->getOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              muxOperand.getDefiningOp()))
        if (branches.contains(cast<handshake::ConditionalBranchOp>(
                muxOperand.getDefiningOp()))) {
          secondFoundCycle = true;
          secondMuxCycleInputIdx = operIdx;
          secondCondBranchOp =
              cast<handshake::ConditionalBranchOp>(muxOperand.getDefiningOp());
          break;
        }
      operIdx++;
    }
    if (!secondFoundCycle)
      return failure();
    int secondMuxOuterInputIdx = (secondMuxCycleInputIdx == 0) ? 1 : 0;

    // The secondCondBranchOp should not have any more successors; otherwise,
    // it is not a Repeat structure
    if (std::distance(secondCondBranchOp->getResults().getUsers().begin(),
                      secondCondBranchOp->getResults().getUsers().end()) != 1)
      return failure();

    OperandRange firstMuxDataOperands = firstMuxOp.getDataOperands();
    Value firstMuxOuterOperand = firstMuxDataOperands[firstMuxOuterInputIdx];
    Value firstMuxInnerOperand = firstMuxDataOperands[firstMuxCycleInputIdx];
    OperandRange secondMuxDataOperands = secondMuxOp.getDataOperands();
    Value secondMuxOuterOperand = secondMuxDataOperands[secondMuxOuterInputIdx];
    Value secondMuxInnerOperand = secondMuxDataOperands[secondMuxCycleInputIdx];

    // Identify which output of the two Branches feeds the muxInnerOperand
    Value firstBranchTrueResult = firstCondBranchOp.getTrueResult();
    Value firstBranchFalseResult = firstCondBranchOp.getFalseResult();
    bool firstTrueIterFlag = (firstBranchTrueResult == firstMuxInnerOperand);
    Value secondBranchTrueResult = secondCondBranchOp.getTrueResult();
    Value secondBranchFalseResult = secondCondBranchOp.getFalseResult();
    bool secondTrueIterFlag = (secondBranchTrueResult == secondMuxInnerOperand);

    Value condBr1 = firstCondBranchOp.getConditionOperand();
    Value condBr2 = secondCondBranchOp.getConditionOperand();
    if (firstTrueIterFlag && !secondTrueIterFlag) {

      // Check if the condition already feeds a NOT, no need to create a new one
      bool foundNot = false;
      handshake::NotOp existingNotOp;
      for (auto condRes : condBr2.getUsers()) {
        if (isa_and_nonnull<handshake::NotOp>(condRes)) {
          foundNot = true;
          existingNotOp = cast<handshake::NotOp>(condRes);
          break;
        }
      }
      Value newCond;
      if (foundNot) {
        newCond = existingNotOp.getResult();
      } else {
        rewriter.setInsertionPoint(secondCondBranchOp);
        handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
            secondCondBranchOp->getLoc(), condBr2);
        inheritBB(secondCondBranchOp, notOp);

        newCond = notOp.getResult();
      }

      rewriter.replaceAllUsesWith(condBr2, newCond);

      rewriter.replaceAllUsesWith(secondBranchFalseResult,
                                  secondBranchTrueResult);
      secondTrueIterFlag = true;

      // Retrieve the new value of the condition, in case it is not updated
      condBr2 = secondCondBranchOp.getConditionOperand();

    } else if (!firstTrueIterFlag && secondTrueIterFlag) {

      // Check if the condition already feeds a NOT, no need to create a new one
      bool foundNot = false;
      handshake::NotOp existingNotOp;
      for (auto condRes : condBr1.getUsers()) {
        if (isa_and_nonnull<handshake::NotOp>(condRes)) {
          foundNot = true;
          existingNotOp = cast<handshake::NotOp>(condRes);
          break;
        }
      }
      Value newCond;
      if (foundNot) {
        newCond = existingNotOp.getResult();
      } else {
        rewriter.setInsertionPoint(firstCondBranchOp);
        // Insert a NOT at the condition input of the first Branch
        handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
            firstCondBranchOp->getLoc(), condBr1);
        inheritBB(firstCondBranchOp, notOp);

        newCond = notOp.getResult();
      }

      rewriter.replaceAllUsesWith(condBr1, newCond);

      // Replace all uses coming from the false side of the second Branch with
      // the true side of it
      rewriter.replaceAllUsesWith(firstBranchFalseResult,
                                  firstBranchTrueResult);
      // Adjust the secondTrueIterFlag
      firstTrueIterFlag = true;

      // Retrieve the new value of the condition, in case it is not updated
      condBr1 = firstCondBranchOp.getConditionOperand();
    }

    int64_t constantValue;
    if (firstTrueIterFlag) {
      assert(secondTrueIterFlag);
      // this means repeat when the condition is true
      constantValue = 1;
    } else {
      assert(!firstTrueIterFlag && !secondTrueIterFlag);
      // this means repeat when the condition is false
      constantValue = 0;
    }
    Value source =
        rewriter.create<handshake::SourceOp>(secondCondBranchOp->getLoc());
    Type constantType = rewriter.getIntegerType(1);
    Value constantVal = rewriter.create<handshake::ConstantOp>(
        secondCondBranchOp->getLoc(), constantType,
        rewriter.getIntegerAttr(constantType, constantValue), source);

    // Create a new Mux and assign its operands
    ValueRange muxOperands;
    if (firstTrueIterFlag) {
      assert(firstTrueIterFlag);
      muxOperands = {condBr1, constantVal};
    } else {
      assert(!firstTrueIterFlag && !firstTrueIterFlag);
      muxOperands = {constantVal, condBr1};
    }
    rewriter.setInsertionPoint(secondCondBranchOp);
    handshake::MuxOp mux = rewriter.create<handshake::MuxOp>(
        secondCondBranchOp->getLoc(), condBr2, muxOperands);
    inheritBB(secondCondBranchOp, mux);

    Value muxResult = mux.getResult();

    assert(isa_and_nonnull<handshake::MergeOp>(
        secondMuxOp.getSelectOperand().getDefiningOp()));
    handshake::MergeOp initOp = cast<handshake::MergeOp>(
        secondMuxOp.getSelectOperand().getDefiningOp());
    rewriter.replaceAllUsesWith(initOp.getDataOperands()[0], muxResult);

    rewriter.replaceAllUsesWith(condBr2, muxResult);

    rewriter.replaceAllUsesWith(secondMuxOuterOperand, firstMuxOuterOperand);

    rewriter.replaceAllUsesWith(firstCondBranchOp.getDataOperand(),
                                firstMuxOuterOperand);
    rewriter.eraseOp(firstMuxOp);
    rewriter.eraseOp(firstCondBranchOp);

    return success();
  }
};

// Replaces a pair of consecutive Repeats with a
// a single Repeat with a merge at its condition input.
struct ShortenMergeRepeatPairs : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp firstMergeOp,
                                PatternRewriter &rewriter) const override {
    auto firstMergeUsers = (firstMergeOp.getResult()).getUsers();
    if (std::distance(firstMergeUsers.begin(), firstMergeUsers.end()) != 2)
      return failure();

    // One user must be a Branch; otherwise, the pattern match fails
    bool firstFoundCondBranch = false;
    handshake::ConditionalBranchOp firstCondBranchOp;
    // One user must be another Merge belonging to a second Repeat; otherwise,
    // the pattern match fails
    bool foundSecondMerge = false;
    handshake::MergeOp secondMergeOp;
    for (auto mergeUser : firstMergeUsers) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(mergeUser)) {
        firstFoundCondBranch = true;
        firstCondBranchOp = cast<handshake::ConditionalBranchOp>(mergeUser);
      } else if (isa_and_nonnull<handshake::MergeOp>(mergeUser)) {
        foundSecondMerge = true;
        secondMergeOp = cast<handshake::MergeOp>(mergeUser);
      }
    }
    if (!firstFoundCondBranch && !foundSecondMerge)
      return failure();

    bool firstFoundCycle = false;
    int operIdx = 0;
    int firstMergeCycleInputIdx = 0;
    for (auto mergeOperand : firstMergeOp->getOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              mergeOperand.getDefiningOp()))
        if (cast<handshake::ConditionalBranchOp>(
                mergeOperand.getDefiningOp()) == firstCondBranchOp) {
          firstFoundCycle = true;
          firstMergeCycleInputIdx = operIdx;
          break;
        }
      operIdx++;
    }
    if (!firstFoundCycle)
      return failure();
    int firstMergeOuterInputIdx = (firstMergeCycleInputIdx == 0) ? 1 : 0;

    // The firstCondBranchOp should not have any more successors; otherwise,
    // it is not a Repeat structure
    if (std::distance(firstCondBranchOp->getResults().getUsers().begin(),
                      firstCondBranchOp->getResults().getUsers().end()) != 1)
      return failure();

    auto secondMergeUsers = (secondMergeOp.getResult()).getUsers();
    if (secondMergeUsers.empty())
      return failure();

    // One user must be a Branch; otherwise, the pattern match fails
    bool secondFoundCondBranch = false;
    // This second Repeat could be feeding many users including maybe another
    // non-loop Branch
    DenseSet<handshake::ConditionalBranchOp> branches;
    for (auto mergeUser : secondMergeUsers) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(mergeUser)) {
        secondFoundCondBranch = true;
        branches.insert(cast<handshake::ConditionalBranchOp>(mergeUser));
      }
    }
    if (!secondFoundCondBranch)
      return failure();

    // One of the branches in the set of Branches must be also be an operand
    // forming a cycle with the merge; otherwise, the pattern match fails
    bool secondFoundCycle = false;
    operIdx = 0;
    int secondMergeCycleInputIdx = 0;
    handshake::ConditionalBranchOp secondCondBranchOp;
    for (auto mergeOperand : secondMergeOp->getOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              mergeOperand.getDefiningOp()))
        if (branches.contains(cast<handshake::ConditionalBranchOp>(
                mergeOperand.getDefiningOp()))) {
          secondFoundCycle = true;
          secondMergeCycleInputIdx = operIdx;
          secondCondBranchOp = cast<handshake::ConditionalBranchOp>(
              mergeOperand.getDefiningOp());
          break;
        }
      operIdx++;
    }
    if (!secondFoundCycle)
      return failure();
    int secondMergeOuterInputIdx = (secondMergeCycleInputIdx == 0) ? 1 : 0;

    if (std::distance(secondCondBranchOp->getResults().getUsers().begin(),
                      secondCondBranchOp->getResults().getUsers().end()) != 1)
      return failure();

    OperandRange firstMergeDataOperands = firstMergeOp.getDataOperands();
    Value firstMergeOuterOperand =
        firstMergeDataOperands[firstMergeOuterInputIdx];
    Value firstMergeInnerOperand =
        firstMergeDataOperands[firstMergeCycleInputIdx];
    OperandRange secondMergeDataOperands = secondMergeOp.getDataOperands();
    Value secondMergeOuterOperand =
        secondMergeDataOperands[secondMergeOuterInputIdx];
    Value secondMergeInnerOperand =
        secondMergeDataOperands[secondMergeCycleInputIdx];

    // Identify which output of the two Branches feeds the mergeInnerOperand
    Value firstBranchTrueResult = firstCondBranchOp.getTrueResult();
    Value firstBranchFalseResult = firstCondBranchOp.getFalseResult();
    bool firstTrueIterFlag = (firstBranchTrueResult == firstMergeInnerOperand);
    Value secondBranchTrueResult = secondCondBranchOp.getTrueResult();
    Value secondBranchFalseResult = secondCondBranchOp.getFalseResult();
    bool secondTrueIterFlag =
        (secondBranchTrueResult == secondMergeInnerOperand);

    Value condBr1 = firstCondBranchOp.getConditionOperand();
    Value condBr2 = secondCondBranchOp.getConditionOperand();
    if (firstTrueIterFlag && !secondTrueIterFlag) {

      // Check if the condition already feeds a NOT, no need to create a new one
      bool foundNot = false;
      handshake::NotOp existingNotOp;
      for (auto condRes : condBr2.getUsers()) {
        if (isa_and_nonnull<handshake::NotOp>(condRes)) {
          foundNot = true;
          existingNotOp = cast<handshake::NotOp>(condRes);
          break;
        }
      }

      Value newCond;
      if (foundNot) {
        newCond = existingNotOp.getResult();
      } else {
        rewriter.setInsertionPoint(secondCondBranchOp);
        // Insert a NOT at the condition input of the second Branch
        handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
            secondCondBranchOp->getLoc(), condBr2);
        inheritBB(secondCondBranchOp, notOp);

        newCond = notOp.getResult();
      }

      rewriter.replaceAllUsesWith(condBr2, newCond);

      rewriter.replaceAllUsesWith(secondBranchFalseResult,
                                  secondBranchTrueResult);
      secondTrueIterFlag = true;

      // Retrieve the new value of the condition, in case it is not updated
      condBr2 = secondCondBranchOp.getConditionOperand();

    } else if (!firstTrueIterFlag && secondTrueIterFlag) {

      // Check if the condition already feeds a NOT, no need to create a new one
      bool foundNot = false;
      handshake::NotOp existingNotOp;
      for (auto condRes : condBr1.getUsers()) {
        if (isa_and_nonnull<handshake::NotOp>(condRes)) {
          foundNot = true;
          existingNotOp = cast<handshake::NotOp>(condRes);
          break;
        }
      }

      Value newCond;
      if (foundNot) {
        newCond = existingNotOp.getResult();
      } else {
        rewriter.setInsertionPoint(firstCondBranchOp);
        // Insert a NOT at the condition input of the first Branch
        handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
            firstCondBranchOp->getLoc(), condBr1);
        inheritBB(firstCondBranchOp, notOp);

        newCond = notOp.getResult();
      }

      rewriter.replaceAllUsesWith(condBr1, newCond);

      // Replace all uses coming from the false side of the second Branch with
      // the true side of it
      rewriter.replaceAllUsesWith(firstBranchFalseResult,
                                  firstBranchTrueResult);
      // Adjust the secondTrueIterFlag
      firstTrueIterFlag = true;

      // Retrieve the new value of the condition, in case it is not updated
      condBr1 = firstCondBranchOp.getConditionOperand();
    }

    int64_t constantValue;
    if (firstTrueIterFlag) {
      assert(secondTrueIterFlag);
      // this means repeat when the condition is true
      constantValue = 1;
    } else {
      assert(!firstTrueIterFlag && !secondTrueIterFlag);
      // this means repeat when the condition is false
      constantValue = 0;
    }
    Value source =
        rewriter.create<handshake::SourceOp>(secondCondBranchOp->getLoc());
    Type constantType = rewriter.getIntegerType(1);
    Value constantVal = rewriter.create<handshake::ConstantOp>(
        secondCondBranchOp->getLoc(), constantType,
        rewriter.getIntegerAttr(constantType, constantValue), source);

    // Create a new Mux and assign its operands
    ValueRange muxOperands;
    if (firstTrueIterFlag) {
      assert(firstTrueIterFlag);
      muxOperands = {condBr1, constantVal};
    } else {
      assert(!firstTrueIterFlag && !firstTrueIterFlag);
      muxOperands = {constantVal, condBr1};
    }
    rewriter.setInsertionPoint(secondCondBranchOp);
    handshake::MuxOp mux = rewriter.create<handshake::MuxOp>(
        secondCondBranchOp->getLoc(), condBr2, muxOperands);
    inheritBB(secondCondBranchOp, mux);

    ////////////////////////////////////////

    Value muxResult = mux.getResult();

    // Correct the condition of the second Branch
    rewriter.replaceAllUsesWith(condBr2, muxResult);

    // Correct the external input of the second Merge
    rewriter.replaceAllUsesWith(secondMergeOuterOperand,
                                firstMergeOuterOperand);

    // Erase the first Branch and first Merge
    rewriter.replaceAllUsesWith(firstCondBranchOp.getDataOperand(),
                                firstMergeOuterOperand);
    rewriter.eraseOp(firstMergeOp);
    rewriter.eraseOp(firstCondBranchOp);

    return success();
  }
};

// Breaks a Branch that has both true and false successors into two
// Suppresses.
struct ConstructSuppresses
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    if (OPTIM_BRANCH_TO_SUPP)
      return failure();
    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();
    if (branchTrueResult.getUsers().empty() ||
        branchFalseResult.getUsers().empty())
      return failure();

    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = condBranchOp.getConditionOperand();

    ValueRange branchOperands = {condOperand, dataOperand};
    rewriter.setInsertionPoint(condBranchOp);
    handshake::ConditionalBranchOp newBranch =
        rewriter.create<handshake::ConditionalBranchOp>(condBranchOp->getLoc(),
                                                        branchOperands);
    inheritBB(condBranchOp, newBranch);

    Value newBranchTrueResult = newBranch.getTrueResult();
    rewriter.replaceAllUsesWith(branchTrueResult, newBranchTrueResult);

    return success();
  }
};

// If a Branch has one successor in the true side, reverse it to be really a
// Suppress
struct FixBranchesToSuppresses
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    if (OPTIM_BRANCH_TO_SUPP)
      return failure();
    // The pattern match fails if the Branch has no true succs or has both
    // true and false succs
    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();
    if (branchTrueResult.getUsers().empty() ||
        (!branchFalseResult.getUsers().empty() &&
         !branchTrueResult.getUsers().empty()))
      return failure();

    // Construct a new Branch that should feed the true side of the old with
    // its false side and takes the inverse of the condition

    // Check if the condition already feeds a NOT, no need to create a new one
    bool foundNot = false;
    handshake::NotOp existingNotOp;
    for (auto condRes : condBranchOp.getConditionOperand().getUsers()) {
      if (isa_and_nonnull<handshake::NotOp>(condRes)) {
        foundNot = true;
        existingNotOp = cast<handshake::NotOp>(condRes);
        break;
      }
    }

    Value condOperand;
    if (foundNot) {
      condOperand = existingNotOp.getResult();

    } else {
      rewriter.setInsertionPoint(condBranchOp);
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          condBranchOp->getLoc(), condBranchOp.getConditionOperand());
      inheritBB(condBranchOp, notOp);

      condOperand = notOp.getResult();
    }

    Value dataOperand = condBranchOp.getDataOperand();

    ValueRange branchOperands = {condOperand, dataOperand};
    rewriter.setInsertionPoint(condBranchOp);
    handshake::ConditionalBranchOp newBranch =
        rewriter.create<handshake::ConditionalBranchOp>(condBranchOp->getLoc(),
                                                        branchOperands);
    inheritBB(condBranchOp, newBranch);

    Value newBranchFalseResult = newBranch.getFalseResult();
    rewriter.replaceAllUsesWith(branchTrueResult, newBranchFalseResult);

    return success();
  }
};

// If a Suppress has two or more successors, feed each successor by a separate
// Suppress
struct DistributeSuppresses
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    if (OPTIM_DISTR)
      return failure();
    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();
    if (!branchTrueResult.getUsers().empty() ||
        branchFalseResult.getUsers().empty())
      return failure();

    // Now, if the Branch has only one user in the false side, there is
    // nothing to be done so the pattern match fails
    if (std::distance(branchFalseResult.getUsers().begin(),
                      branchFalseResult.getUsers().end()) == 1)
      return failure();

    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = condBranchOp.getConditionOperand();

    int numOfUsers = std::distance(branchFalseResult.getUsers().begin(),
                                   branchFalseResult.getUsers().end());
    int i = 0;
    handshake::ConditionalBranchOp oldBranch = condBranchOp;
    while (i < numOfUsers - 1) {
      ValueRange branchOperands = {condOperand, dataOperand};
      rewriter.setInsertionPoint(condBranchOp);
      handshake::ConditionalBranchOp newBranch =
          rewriter.create<handshake::ConditionalBranchOp>(
              condBranchOp->getLoc(), branchOperands);
      inheritBB(condBranchOp, newBranch);

      Value newBranchFalseResult = newBranch.getFalseResult();
      Value branchOldFalseResult = oldBranch.getFalseResult();
      // All users of the old branch will be directed to the new branch except
      // only one
      rewriter.replaceAllUsesExcept(
          branchOldFalseResult, newBranchFalseResult,
          *oldBranch.getFalseResult().getUsers().begin());
      oldBranch = newBranch;
      i++;
    }

    return success();
  }
};

// If a Repeat has two or more successors, feed each successor by a separate
// Repeat
struct DistributeMergeRepeats : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {

    if (OPTIM_DISTR)
      return failure();
    auto mergeUsers = (mergeOp.getResult()).getUsers();
    if (std::distance(mergeUsers.begin(), mergeUsers.end()) < 3)
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
    // mux; otherwise, the pattern match fails
    bool foundCycle = false;
    int operIdx = 0;
    int mergeOuterInputIdx = 0;
    int mergeCycleInputIdx = 0;
    handshake::ConditionalBranchOp condBranchOp;
    for (auto mergeOperand : mergeOp.getDataOperands()) {
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
    mergeOuterInputIdx = (mergeCycleInputIdx) ? 0 : 1;

    if (!condBranchOp.getTrueResult().getUsers().empty() ||
        condBranchOp.getFalseResult().getUsers().empty())
      return failure();

    int numOfUsers = std::distance(mergeUsers.begin(), mergeUsers.end()) - 1;

    int i = 0;
    handshake::MergeOp oldMergeOp = mergeOp;
    handshake::ConditionalBranchOp oldBranchOp = condBranchOp;
    Value mergeOuterInput = mergeOp.getDataOperands()[mergeOuterInputIdx];
    Value branchCond = condBranchOp.getConditionOperand();

    while (i < numOfUsers - 1) {
      ValueRange branchOperands = {branchCond, condBranchOp.getFalseResult()};
      rewriter.setInsertionPoint(condBranchOp);
      handshake::ConditionalBranchOp newBranch =
          rewriter.create<handshake::ConditionalBranchOp>(
              condBranchOp->getLoc(), branchOperands);
      inheritBB(condBranchOp, newBranch);

      ValueRange mergeOperands;
      if (mergeOuterInputIdx == 0)
        mergeOperands = {mergeOuterInput, newBranch.getFalseResult()};
      else
        mergeOperands = {newBranch.getFalseResult(), mergeOuterInput};
      rewriter.setInsertionPoint(oldMergeOp);
      handshake::MergeOp newMergeOp = rewriter.create<handshake::MergeOp>(
          oldMergeOp.getLoc(), mergeOperands);
      inheritBB(oldMergeOp, newMergeOp);

      newBranch->setOperand(1, newMergeOp.getResult());

      Value oldMergeResult = oldMergeOp.getResult();
      Value newMergeResult = newMergeOp.getResult();
      rewriter.replaceAllUsesExcept(oldMergeResult, newMergeResult,
                                    oldBranchOp);

      Operation *oneUser;
      for (auto newMergeUser : newMergeOp.getResult().getUsers()) {
        if (newMergeUser != newBranch) {
          oneUser = newMergeUser;
          break;
        }
      }
      int idx = 0;
      for (auto oneUserOperand : oneUser->getOperands()) {
        if (oneUserOperand == newMergeOp)
          break;
        idx++;
      }
      oneUser->setOperand(idx, oldMergeResult);

      oldMergeOp = newMergeOp;
      oldBranchOp = newBranch;
      i++;
    }

    return success();
  }
};

// If a Repeat has two or more successors, feed each successor by a separate
// Repeat
struct DistributeMuxRepeats : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    if (OPTIM_DISTR)
      return failure();

    auto muxUsers = (muxOp.getResult()).getUsers();
    if (std::distance(muxUsers.begin(), muxUsers.end()) < 3)
      return failure();

    // One user must be a Branch; otherwise, the pattern match fails
    bool foundCondBranch = false;
    DenseSet<handshake::ConditionalBranchOp> branches;
    for (auto muxUser : muxUsers) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(muxUser)) {
        foundCondBranch = true;
        branches.insert(cast<handshake::ConditionalBranchOp>(muxUser));
      }
    }
    if (!foundCondBranch)
      return failure();

    // This condBranchOp must also be an operand forming a cycle with the
    // mux; otherwise, the pattern match fails
    bool foundCycle = false;
    int operIdx = 0;
    int muxOuterInputIdx = 0;
    int muxCycleInputIdx = 0;
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
    if (!foundCycle)
      return failure();
    muxOuterInputIdx = (muxCycleInputIdx) ? 0 : 1;

    if (!condBranchOp.getTrueResult().getUsers().empty() ||
        condBranchOp.getFalseResult().getUsers().empty())
      return failure();

    int numOfUsers = std::distance(muxUsers.begin(), muxUsers.end()) - 1;

    int i = 0;
    handshake::MuxOp oldMuxOp = muxOp;
    handshake::ConditionalBranchOp oldBranchOp = condBranchOp;
    Value muxOuterInput = muxOp.getDataOperands()[muxOuterInputIdx];
    Value muxSel = muxOp.getSelectOperand();
    Value branchCond = condBranchOp.getConditionOperand();

    while (i < numOfUsers - 1) {
      ValueRange branchOperands = {branchCond, condBranchOp.getFalseResult()};
      rewriter.setInsertionPoint(condBranchOp);
      handshake::ConditionalBranchOp newBranch =
          rewriter.create<handshake::ConditionalBranchOp>(
              condBranchOp->getLoc(), branchOperands);
      inheritBB(condBranchOp, newBranch);

      ValueRange muxOperands;
      if (muxOuterInputIdx == 0)
        muxOperands = {muxOuterInput, newBranch.getFalseResult()};
      else
        muxOperands = {newBranch.getFalseResult(), muxOuterInput};
      rewriter.setInsertionPoint(oldMuxOp);
      handshake::MuxOp newMuxOp = rewriter.create<handshake::MuxOp>(
          oldMuxOp.getLoc(), muxSel, muxOperands);
      inheritBB(oldMuxOp, newMuxOp);

      newBranch->setOperand(1, newMuxOp.getResult());

      Value oldMergeResult = oldMuxOp.getResult();
      Value newMergeResult = newMuxOp.getResult();
      rewriter.replaceAllUsesExcept(oldMergeResult, newMergeResult,
                                    oldBranchOp);
      Operation *oneUser;
      for (auto newMergeUser : newMuxOp.getResult().getUsers()) {
        if (newMergeUser != newBranch) {
          oneUser = newMergeUser;
          break;
        }
      }
      int idx = 0;
      for (auto oneUserOperand : oneUser->getOperands()) {
        if (oneUserOperand == newMuxOp)
          break;
        idx++;
      }
      oneUser->setOperand(idx, oldMergeResult);

      oldMuxOp = newMuxOp;
      oldBranchOp = newBranch;
      i++;
    }
    return success();
  }
};

struct ConvertLoopMergeToMux : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
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

    if (!OPTIM_BRANCH_TO_SUPP) {
      if (!condBranchOp.getTrueResult().getUsers().empty() ||
          condBranchOp.getFalseResult().getUsers().empty())
        return failure();
    }

    mergeOuterInputIdx = (mergeCycleInputIdx == 0) ? 1 : 0;

    OperandRange mergeDataOperands = mergeOp.getDataOperands();
    Value mergeInnerOperand = mergeDataOperands[mergeCycleInputIdx];

    bool isTrueOutputOuter = false;
    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();
    Value branchOuterResult;
    if (branchTrueResult == mergeInnerOperand)
      branchOuterResult = branchFalseResult;
    else if (branchFalseResult == mergeInnerOperand) {
      branchOuterResult = branchTrueResult;
      isTrueOutputOuter = true;
    } else
      return failure();

    Value condition = condBranchOp.getConditionOperand();
    bool needNot = ((isTrueOutputOuter && mergeCycleInputIdx == 1) ||
                    (!isTrueOutputOuter && mergeCycleInputIdx == 0));
    Value iterCond;
    if (needNot) {
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
      } else {
        rewriter.setInsertionPoint(condBranchOp);
        handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
            condBranchOp->getLoc(), condition);
        inheritBB(condBranchOp, notOp);
        iterCond = notOp.getResult();
      }

    } else {
      iterCond = condition;
    }

    int constVal = mergeOuterInputIdx;
    // Obtain the start signal from the last argument of any block
    Block *mergeBlock = mergeOp->getBlock();
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
    } else {
      // Create a new ConstantOp in the same block as that of the branch
      // forming the cycle
      Type constantType = rewriter.getIntegerType(1);
      rewriter.setInsertionPoint(mergeOp);
      Value valueOfConstant = rewriter.create<handshake::ConstantOp>(
          mergeOp->getLoc(), constantType,
          rewriter.getIntegerAttr(constantType, constVal), start);

      ValueRange operands = {iterCond, valueOfConstant};
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

    return success();
  }
};

/// Simple driver for the Handshake Rewrite Terms pass, based on a greedy
/// pattern rewriter.
struct HandshakeRewriteTermsPass
    : public dynamatic::impl::HandshakeRewriteTermsBase<
          HandshakeRewriteTermsPass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns(ctx);
    patterns.add<
        EraseUnconditionalBranches, EraseSingleInputMerges,
        EraseSingleInputMuxes, EraseSingleInputControlMerges,
        DowngradeIndexlessControlMerge, RemoveDoubleSinkBranches,
        RemoveMuxFloatingLoop, RemoveMergeFloatingLoop, ConstructSuppresses,
        FixBranchesToSuppresses, DistributeSuppresses, DistributeMergeRepeats,
        DistributeMuxRepeats, ExtractIfThenElseCondition, ExtractLoopCondition,
        RemoveBranchMergeIfThenElse, RemoveBranchMuxIfThenElse,
        RemoveMergeBranchLoop, RemoveMuxBranchLoop, ConvertLoopMergeToMux>(ctx);

    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::rewriteHandshakeTerms() {
  return std::make_unique<HandshakeRewriteTermsPass>();
}
