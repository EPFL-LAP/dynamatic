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
#include <cassert>
#include <iterator>
#include <vector>

using namespace mlir;
using namespace dynamatic;

namespace {

// Rules E
/// Erases unconditional branches (which would eventually lower to simple
/// wires).
struct EraseUnconditionalBranches
    : public OpRewritePattern<handshake::BranchOp> {
  using OpRewritePattern<handshake::BranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::BranchOp brOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(brOp, brOp.getDataOperand());
    // llvm::errs() << "\t***Rules E: Removing unconditional Branch!!***\n";
    return success();
  }
};

// Rules E
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

// Rules E
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

// Rules E
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

// Rules E
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

// Rules E
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
    llvm::errs() << "\t***Rules E: remove-double-sink-branch!***\n";

    return success();
  }
};

// Rules E
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

    llvm::errs() << "\t\tDeleting isolated cycle\n";
    rewriter.replaceAllUsesWith(iterCondBranchOp.getDataOperand(),
                                mergeOp.getDataOperands()[mergeOuterInputIdx]);
    rewriter.eraseOp(mergeOp);
    rewriter.eraseOp(iterCondBranchOp);

    llvm::errs() << "\t***Rules E: remove-floating-merge-loop!***\n";

    return success();
  }
};

// Rules E
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

    llvm::errs() << "\t\tDeleting isolated cycle\n";
    rewriter.replaceAllUsesWith(iterCondBranchOp.getDataOperand(),
                                muxOp.getDataOperands()[muxOuterInputIdx]);
    rewriter.eraseOp(muxOp);
    rewriter.eraseOp(iterCondBranchOp);

    llvm::errs() << "\t***Rules E: remove-floating-merge-loop!***\n";

    return success();
  }
};

// TODO: In its current form, the rule will apply even if the Branches has more
// succs than the Merge. It is not wrong given that we are implement the
// Supp->Fork distribute rule
// Rules A Removes Conditional Branch and Merge
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

    // The two operands of the Cmerge should be conditional Branches; otherwise,
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

    // If the only user of the two Branches is the merge, delete them
    if ((std::distance(firstBranchOperand.getTrueResult().getUsers().begin(),
                       firstBranchOperand.getTrueResult().getUsers().end()) +
         std::distance(firstBranchOperand.getFalseResult().getUsers().begin(),
                       firstBranchOperand.getFalseResult().getUsers().end())) ==
        1)
      rewriter.eraseOp(firstBranchOperand);

    if ((std::distance(secondBranchOperand.getTrueResult().getUsers().begin(),
                       secondBranchOperand.getTrueResult().getUsers().end()) +
         std::distance(
             secondBranchOperand.getFalseResult().getUsers().begin(),
             secondBranchOperand.getFalseResult().getUsers().end())) == 1)
      rewriter.eraseOp(secondBranchOperand);

    llvm::errs() << "\t***Rules A: remove-branch-merge-if-then-else!***\n";
    return success();
  }
};

// TODO: In its current form, the rule will apply even if the Branches has more
// succs than the Merge. It is not wrong given that we are implement the
// Supp->Fork distribute rule
// Rules A Removes Conditional Branch and Mux
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

    // Delegated the deletiong to such Branches through a separate function that
    // deletes Branches ffedng sinks on both sides
    // If the only user of the two
    // Branches is the merge, delete them if
    // ((std::distance(firstBranchOperand.getTrueResult().getUsers().begin(),
    //                    firstBranchOperand.getTrueResult().getUsers().end()) +
    //      std::distance(firstBranchOperand.getFalseResult().getUsers().begin(),
    //                    firstBranchOperand.getFalseResult().getUsers().end()))
    //                    ==
    //     1)
    //   rewriter.eraseOp(firstBranchOperand);

    // if
    // ((std::distance(secondBranchOperand.getTrueResult().getUsers().begin(),
    //                    secondBranchOperand.getTrueResult().getUsers().end())
    //                    +
    //      std::distance(
    //          secondBranchOperand.getFalseResult().getUsers().begin(),
    //          secondBranchOperand.getFalseResult().getUsers().end())) == 1)
    //   rewriter.eraseOp(secondBranchOperand);

    llvm::errs() << "\t***Rules A: remove-branch-mux-if-then-else!***\n";
    return success();
  }
};

// TODO: In its current form, the rule will apply even if the Mux has more
// succs than the Branches. It is not wrong given that we are implement the
// Repeat->Fork distribute rule
// Rules A
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

    llvm::errs() << "\t\t(1) Found a Merge-Branch loop\n";

    // One of the branches in the set of Branches must have the same condition
    // as that of iterCondBranchOp but has none of its succs equal to the merge
    Value iterBranchCondition = iterCondBranchOp.getConditionOperand();
    Value iterOriginalBranchCondition = iterBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(iterBranchCondition.getDefiningOp()))
      iterOriginalBranchCondition =
          iterBranchCondition.getDefiningOp()->getOperand(0);

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

      if (brOriginalBranchCondition == iterOriginalBranchCondition) {
        Value brTrueResult = br.getTrueResult();
        Value brFalseResult = br.getFalseResult();
        // Make sure that this Branch is not making a cycle with the Merge
        if (brTrueResult != mergeOp.getDataOperands()[mergeCycleInputIdx] &&
            brFalseResult != mergeOp.getDataOperands()[mergeCycleInputIdx] &&
            brTrueResult != mergeOp.getDataOperands()[mergeOuterInputIdx] &&
            brFalseResult != mergeOp.getDataOperands()[mergeOuterInputIdx]) {
          foundExitBranch = true;
          exitCondBranchOp = br;
          exitBranchCondition = brBranchCondition;
          exitOriginalBranchCondition = brOriginalBranchCondition;
          break;
        }
      }
    }

    if (!foundExitBranch)
      return failure();

    llvm::errs() << "\t\t(2) Found an exit Branch\n";

    // This exitCondBranchOp has to be a suppress with at least one succ in only
    // one side; otherwise, the pattern match fails
    if ((!exitCondBranchOp.getTrueResult().getUsers().empty() &&
         !exitCondBranchOp.getFalseResult().getUsers().empty()) ||
        (exitCondBranchOp.getTrueResult().getUsers().empty() &&
         exitCondBranchOp.getFalseResult().getUsers().empty()))
      return failure();

    llvm::errs() << "\t\t(3) Exit Branch is a Suppress\n";

    // The two branches must be steering oppositely: Either have the same exact
    // condition but opposite result signs OR opposite conditions and same
    // result signs; otherwise, if they are steering with the same sign, the
    // pattern match fails
    bool firstPassWithTrueCond =
        ((iterCondBranchOp.getTrueResult() ==
              mergeOp.getDataOperands()[mergeCycleInputIdx] &&
          iterBranchCondition == iterOriginalBranchCondition) ||
         (iterCondBranchOp.getFalseResult() ==
              mergeOp.getDataOperands()[mergeCycleInputIdx] &&
          iterBranchCondition != iterOriginalBranchCondition));

    bool secondPassWithTrueCond =
        ((!exitCondBranchOp.getTrueResult().getUsers().empty() &&
          exitBranchCondition == exitOriginalBranchCondition) ||
         (!exitCondBranchOp.getFalseResult().getUsers().empty() &&
          exitBranchCondition != exitOriginalBranchCondition));
    if (secondPassWithTrueCond == firstPassWithTrueCond)
      return failure();

    llvm::errs()
        << "\t\t(3) Exit Branch and Loop Branch haave opposite signs\n";

    // Replace all uses of the branchOuterResult with
    // the muxOuterOperand
    Value brTrueResult = exitCondBranchOp.getTrueResult();
    if (!brTrueResult.getUsers().empty())
      rewriter.replaceAllUsesWith(
          brTrueResult, mergeOp.getDataOperands()[mergeOuterInputIdx]);
    else {
      Value brFalseResult = exitCondBranchOp.getFalseResult();
      assert(!brFalseResult.getUsers().empty());
      rewriter.replaceAllUsesWith(
          brFalseResult, mergeOp.getDataOperands()[mergeOuterInputIdx]);
    }

    // llvm::errs() << "\t\t(4) Found a Merge-Branch loop";

    // Decided on delegating this to another function that is reponsible for
    // deleting floating cycles
    // If the only user of the merge output are the 2 Branches AND the only
    // user of the iterCondBranchOp's iterator output is the merge, delete both
    // of them
    // llvm::errs()
    //     << "DISTANCES: " << std::distance(mergeUsers.begin(),
    //     mergeUsers.end())
    //     << ", "
    //     <<
    //     (std::distance(iterCondBranchOp.getTrueResult().getUsers().begin(),
    //                       iterCondBranchOp.getTrueResult().getUsers().end())
    //                       +
    //         std::distance(iterCondBranchOp.getFalseResult().getUsers().begin(),
    //                       iterCondBranchOp.getFalseResult().getUsers().end()))
    //     << "\n";
    // if (std::distance(mergeUsers.begin(), mergeUsers.end()) < 3 &&
    //     ((std::distance(iterCondBranchOp.getTrueResult().getUsers().begin(),
    //                     iterCondBranchOp.getTrueResult().getUsers().end()) +
    //       std::distance(iterCondBranchOp.getFalseResult().getUsers().begin(),
    //                     iterCondBranchOp.getFalseResult().getUsers().end()))
    //                     ==
    //      1)) {
    //   llvm::errs() << "\t\tSHOULD DELETE!!\n";
    //   rewriter.replaceAllUsesWith(
    //       iterCondBranchOp.getDataOperand(),
    //       mergeOp.getDataOperands()[mergeOuterInputIdx]);
    //   rewriter.eraseOp(mergeOp);
    //   rewriter.eraseOp(iterCondBranchOp);
    // }

    // We rely on the RemoveUselessBranches to remove exitCondBranchOp if the
    // only user of the exitCondBranchOp is the Mux

    llvm::errs() << "\t***Rules A: remove-merge-branch-loop!***\n";

    return success();
  }
};

// TODO: In its current form, the rule will apply even if the Mux has more
// succs than the Branches. It is not wrong given that we are implement the
// Repeat->Fork distribute rule
// Rules A
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

    // llvm::errs() << "\t\t(1) Found a Merge-Branch loop";

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

    // One of the branches in the set of Branches must have the same condition
    // as that of iterCondBranchOp but has none of its succs equal to the Cmerge
    Value iterBranchCondition = iterCondBranchOp.getConditionOperand();
    Value iterOriginalBranchCondition = iterBranchCondition;
    if (isa_and_nonnull<handshake::NotOp>(iterBranchCondition.getDefiningOp()))
      iterOriginalBranchCondition =
          iterBranchCondition.getDefiningOp()->getOperand(0);

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

      if (brOriginalBranchCondition == iterOriginalBranchCondition) {
        Value brTrueResult = br.getTrueResult();
        Value brFalseResult = br.getFalseResult();
        // Make sure that this Branch is not making a cycle with the Merge
        if (brTrueResult != muxOp.getDataOperands()[muxCycleInputIdx] &&
            brFalseResult != muxOp.getDataOperands()[muxCycleInputIdx] &&
            brTrueResult != muxOp.getDataOperands()[muxOuterInputIdx] &&
            brFalseResult != muxOp.getDataOperands()[muxOuterInputIdx]) {
          foundExitBranch = true;
          exitCondBranchOp = br;
          exitBranchCondition = brBranchCondition;
          exitOriginalBranchCondition = brOriginalBranchCondition;
          break;
        }
      }
    }

    if (!foundExitBranch)
      return failure();

    // This exitCondBranchOp has to be a suppress with at least one succ in only
    // one side; otherwise, the pattern match fails
    if ((!exitCondBranchOp.getTrueResult().getUsers().empty() &&
         !exitCondBranchOp.getFalseResult().getUsers().empty()) ||
        (exitCondBranchOp.getTrueResult().getUsers().empty() &&
         exitCondBranchOp.getFalseResult().getUsers().empty()))
      return failure();

    // The two branches must be steering oppositely: Either have the same exact
    // condition but opposite result signs OR opposite conditions and same
    // result signs; otherwise, if they are steering with the same sign, the
    // pattern match fails
    bool firstPassWithTrueCond =
        ((iterCondBranchOp.getTrueResult() ==
              muxOp.getDataOperands()[muxCycleInputIdx] &&
          iterBranchCondition == iterOriginalBranchCondition) ||
         (iterCondBranchOp.getFalseResult() ==
              muxOp.getDataOperands()[muxCycleInputIdx] &&
          iterBranchCondition != iterOriginalBranchCondition));

    bool secondPassWithTrueCond =
        ((!exitCondBranchOp.getTrueResult().getUsers().empty() &&
          exitBranchCondition == exitOriginalBranchCondition) ||
         (!exitCondBranchOp.getFalseResult().getUsers().empty() &&
          exitBranchCondition != exitOriginalBranchCondition));

    if (secondPassWithTrueCond == firstPassWithTrueCond)
      return failure();

    // Replace all uses of the branchOuterResult with
    // the muxOuterOperand
    Value brTrueResult = exitCondBranchOp.getTrueResult();
    if (!brTrueResult.getUsers().empty())
      rewriter.replaceAllUsesWith(brTrueResult,
                                  muxOp.getDataOperands()[muxOuterInputIdx]);
    else {
      Value brFalseResult = exitCondBranchOp.getFalseResult();
      assert(!brFalseResult.getUsers().empty());
      rewriter.replaceAllUsesWith(brFalseResult,
                                  muxOp.getDataOperands()[muxOuterInputIdx]);
    }

    // llvm::errs() << "\t\t(4) Found a Merge-Branch loop";

    // Decided on delegating this to another function that is reponsible for
    // deleting floating cycles
    // If the only user of the merge output are the 2
    // Branches AND the only user of the iterCondBranchOp's iterator output is
    // the merge, delete both of them llvm::errs()
    //     << "DISTANCES: " << std::distance(muxUsers.begin(), muxUsers.end())
    //     << ", "
    //     <<
    //     (std::distance(iterCondBranchOp.getTrueResult().getUsers().begin(),
    //                       iterCondBranchOp.getTrueResult().getUsers().end())
    //                       +
    //         std::distance(iterCondBranchOp.getFalseResult().getUsers().begin(),
    //                       iterCondBranchOp.getFalseResult().getUsers().end()))
    //     << "\n";
    // if (std::distance(muxUsers.begin(), muxUsers.end()) < 3 &&
    //     ((std::distance(iterCondBranchOp.getTrueResult().getUsers().begin(),
    //                     iterCondBranchOp.getTrueResult().getUsers().end()) +
    //       std::distance(iterCondBranchOp.getFalseResult().getUsers().begin(),
    //                     iterCondBranchOp.getFalseResult().getUsers().end()))
    //                     ==
    //      1)) {
    //   llvm::errs() << "\t\tSHOULD DELETE!!\n";
    //   rewriter.replaceAllUsesWith(iterCondBranchOp.getDataOperand(),
    //                               muxOp.getDataOperands()[muxOuterInputIdx]);
    //   rewriter.eraseOp(muxOp);
    //   rewriter.eraseOp(iterCondBranchOp);
    // }

    // We rely on the RemoveUselessBranches to remove exitCondBranchOp if the
    // only user of the exitCondBranchOp is the Mux

    llvm::errs() << "\t***Rules A: remove-mux-branch-loop!***\n";

    return success();
  }
};

// Rules B
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

    // Replace all uses of the cmerge index with an INIT
    // 1st) Identify whether the loop condition will be connected directly or
    // through a NOT
    // Note: This strategy is correct, but might result in the insertion of
    Value condition = condBranchOp.getConditionOperand();
    bool needNot = ((isTrueOutputOuter && cmergeCycleInputIdx == 1) ||
                    (!isTrueOutputOuter && cmergeCycleInputIdx == 0));
    Value iterCond;
    if (needNot) {
      handshake::NotOp notOp =
          rewriter.create<handshake::NotOp>(condBranchOp->getLoc(), condition);
      iterCond = notOp.getResult();
    } else {
      iterCond = condition;
    }

    // 2nd) Identify the value of the constant that will be triggered from Start
    // and add it
    // The value of the constant should be the cmergeOuterInputIdx
    int constVal = cmergeOuterInputIdx;
    // Obtain the start signal from the last argument of any block
    Block *cmergeBlock = cmergeOp->getBlock();
    MutableArrayRef<BlockArgument> l = cmergeBlock->getArguments();
    if (l.empty())
      return failure();
    mlir::Value start = l.back();
    if (!isa<NoneType>(start.getType()))
      return failure();

    // Create a new ConstantOp in the same block as that of the branch
    // forming the cycle
    Type constantType = rewriter.getIntegerType(1);
    Value valueOfConstant = rewriter.create<handshake::ConstantOp>(
        condBranchOp->getLoc(), constantType,
        rewriter.getIntegerAttr(constantType, constVal), start);

    // 3rd) Add a new Merge operation to serve as the INIT
    ValueRange operands = {iterCond, valueOfConstant};
    handshake::MergeOp mergeOp =
        rewriter.create<handshake::MergeOp>(cmergeOp.getLoc(), operands);
    Value muxSel = mergeOp.getResult();

    Value index = cmergeOp.getIndex();
    rewriter.replaceAllUsesWith(index, muxSel);

    llvm::errs() << "\t***Rules B: extract-loop-mux-condition!***\n";
    return success();
  }
};

// Rules B
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

    // Check if we need to negate the condition before feeding it to the index
    // output of the cmerge
    // (1) Should negate if the in0 receives the true succ of the Branch and the
    // condition of the Branch is not negated OR if it receives the false succ
    // and the condition of the Branch is negated
    bool reversedFirstInput =
        (firstBranchOperand.getTrueResult() == cmergeOp.getOperands()[0] &&
         firstBranchCondition == firstOriginalBranchCondition) ||
        (firstBranchOperand.getFalseResult() == cmergeOp.getOperands()[0] &&
         firstBranchCondition != firstOriginalBranchCondition);
    // (1) Should negate if the in0 receives the true succ of the Branch and the
    // condition of the Branch is not negated OR if it receives the false succ
    // and the condition of the Branch is negated
    bool reversedSecondInput =
        (secondBranchOperand.getFalseResult() == cmergeOp.getOperands()[1] &&
         secondBranchCondition == secondOriginalBranchCondition) ||
        (secondBranchOperand.getTrueResult() == cmergeOp.getOperands()[1] &&
         secondBranchCondition != secondOriginalBranchCondition);

    bool needNot = reversedFirstInput && reversedSecondInput;
    Value cond;
    if (needNot) {
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          cmergeOp->getLoc(), firstOriginalBranchCondition);
      cond = notOp.getResult();
    } else {
      cond = firstOriginalBranchCondition;
    }

    // Replace the Cmerge index output with the branch condition
    rewriter.replaceAllUsesWith(index, cond);

    llvm::errs() << "\t***Rules B: extract-if-then-else-mux-condition!***\n";
    return success();
  }
};

// Rules C
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

    // For the shortening to work, the two branches should have their
    // successor in the same direction (either true or false); otherwise, we
    // need to enforce it by negating.. When they are not consistent, we will
    // force both to have their succs in the false side and sink in true side
    // (like a typical suppress)
    Value condBr1 = firstCondBranchOp.getConditionOperand();
    Value condBr2 = secondCondBranchOp.getConditionOperand();
    if (firstTrueSuccOnlyFlag && secondFalseSuccOnlyFlag) {
      // Insert a NOT at the condition input of the first Branch
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          firstCondBranchOp->getLoc(), condBr1);
      Value newCond = notOp.getResult();
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
      llvm::errs() << firstFalseSuccOnlyFlag << ", " << secondTrueSuccOnlyFlag
                   << ", " << firstTrueSuccOnlyFlag << ", "
                   << secondFalseSuccOnlyFlag << "\n";
      assert(firstFalseSuccOnlyFlag && secondTrueSuccOnlyFlag);
      // Insert a NOT at the condition input of the second Branch
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          secondCondBranchOp->getLoc(), condBr2);
      Value newCond = notOp.getResult();
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

    // The goal now is to replace the two Branches with a single Branch, we do
    // so by deleting the first branch and adjusting the inputs of the second
    // branch
    // The new condition is a Mux, calculate its inputs: One input of the
    // Mux will be a constant that should take the value of the condition that
    // feeds a sink (for suppressing) and should be triggered from Source
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
      // This means suppress when the condition is false, so put the constVal
      // at in0 and the additional condition at in1
      muxOperands = {constantVal, condBr2};
    } else {
      assert(firstFalseSuccOnlyFlag && secondFalseSuccOnlyFlag);
      // This means suppress when the condition is true, so put the constVal
      // at in1 and the additional condition at in0
      muxOperands = {condBr2, constantVal};
    }
    rewriter.setInsertionPoint(secondCondBranchOp);
    handshake::MuxOp mux = rewriter.create<handshake::MuxOp>(
        secondCondBranchOp->getLoc(), condBr1, muxOperands);

    // Correct the inputs of the second Branch
    Value muxResult = mux.getResult();
    Value dataOperand = firstCondBranchOp.getDataOperand();
    ValueRange branchOperands = {muxResult, dataOperand};
    secondCondBranchOp->setOperands(branchOperands);

    // Erase the first Branch
    rewriter.eraseOp(firstCondBranchOp);

    llvm::errs() << "\t***Rules C: shorten-suppress-pairs!***\n";

    return success();
  }
};

// Rules C
// Replaces a pair of consecutive Repeats with a
// a single Repeat with a mux at its condition input.
struct ShortenMuxRepeatPairs : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp firstMuxOp,
                                PatternRewriter &rewriter) const override {
    // Search for a Repeat structure
    // (1) Get the users of the Mux. If they are not exactly two, the pattern
    // match fails
    auto firstMuxUsers = (firstMuxOp.getResult()).getUsers();
    if (std::distance(firstMuxUsers.begin(), firstMuxUsers.end()) != 2)
      return failure();

    // If the mux is not driven by a Merge (i.e., INIT), the pattern match
    // fails
    if (!isa_and_nonnull<handshake::MergeOp>(
            firstMuxOp.getSelectOperand().getDefiningOp()))
      return failure();

    // One user must be a Branch; otherwise, the pattern match fails
    bool firstFoundCondBranch = false;
    handshake::ConditionalBranchOp firstCondBranchOp;
    // Also, One user must be another Mux belonging to a second Repeat;
    // otherwise, the pattern match fails
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

    // At this point we have firstMuxOp and firstCondBranchOp which constitute
    // the first Repeat sturcture. It should feed a second Repeat structure
    // otherwise the pattern match fails
    // Check if secondMuxOp also has a Branch forming a cycle
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

    // Now, we are sure we have two consecutive Repeats, check the signs of
    // loop conditions. Retrieve the values at the Muxes inputs Retrieve the
    // values at the mux inputs
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
      // Insert a NOT at the condition input of the second Branch
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          secondCondBranchOp->getLoc(), condBr2);
      Value newCond = notOp.getResult();
      rewriter.replaceAllUsesWith(condBr2, newCond);

      // Replace all uses coming from the false side of the second Branch with
      // the true side of it
      rewriter.replaceAllUsesWith(secondBranchFalseResult,
                                  secondBranchTrueResult);
      // Adjust the secondTrueIterFlag
      secondTrueIterFlag = true;

      // Retrieve the new value of the condition, in case it is not updated
      condBr2 = secondCondBranchOp.getConditionOperand();

    } else if (!firstTrueIterFlag && secondTrueIterFlag) {
      // Insert a NOT at the condition input of the first Branch
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          firstCondBranchOp->getLoc(), condBr1);
      Value newCond = notOp.getResult();
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

    // The goal now is to replace the two Repeats with a single Repeat, we do
    // so by deleting the first Mux and Branch and adjusting the inputs of the
    // second Mux The new condition is a Mux, calculate its inputs: One input
    // of the Mux will be a constant that should take the value of the
    // condition that feeds a sink (for suppressing) and should be triggered
    // from Source
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
      // This means repeat when the condition is true, so put the constVal at
      // in1 and the additional condition (i.e., condition of the first
      // Repeat) at in0
      muxOperands = {condBr1, constantVal};
    } else {
      assert(!firstTrueIterFlag && !firstTrueIterFlag);
      // This means repeat when the condition is false, so put the constVal at
      // in0 and the additional condition (i.e., the condition of the first
      // Repeat) at in1
      muxOperands = {constantVal, condBr1};
    }
    rewriter.setInsertionPoint(secondCondBranchOp);
    handshake::MuxOp mux = rewriter.create<handshake::MuxOp>(
        secondCondBranchOp->getLoc(), condBr2, muxOperands);
    Value muxResult = mux.getResult();

    // Correct the select of the second Mux; at this point, we are sure it
    // comes from a Merge (INIT), so retrieve it
    assert(isa_and_nonnull<handshake::MergeOp>(
        secondMuxOp.getSelectOperand().getDefiningOp()));
    handshake::MergeOp initOp = cast<handshake::MergeOp>(
        secondMuxOp.getSelectOperand().getDefiningOp());
    // The convention used in the ExtractLoopMuxCondition rewrite puts the
    // loop condition at in0 of the Merge
    rewriter.replaceAllUsesWith(initOp.getDataOperands()[0], muxResult);

    // Correct the condition of the second Branch
    rewriter.replaceAllUsesWith(condBr2, muxResult);

    // Correct the external input of the second Mux
    rewriter.replaceAllUsesWith(secondMuxOuterOperand, firstMuxOuterOperand);

    // Erase the first Branch and first Mux
    rewriter.replaceAllUsesWith(firstCondBranchOp.getDataOperand(),
                                firstMuxOuterOperand);
    rewriter.eraseOp(firstMuxOp);
    rewriter.eraseOp(firstCondBranchOp);

    // TODO: Erase the first INIT as well!!!!!

    llvm::errs() << "\t***Rules C: shorten-mux-repeat-pairs!***\n";

    return success();
  }
};

// Rules C
// Replaces a pair of consecutive Repeats with a
// a single Repeat with a merge at its condition input.
struct ShortenMergeRepeatPairs : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp firstMergeOp,
                                PatternRewriter &rewriter) const override {
    // Search for a Repeat structure
    // (1) Get the users of the Merge. If they are not exactly two, the
    // pattern match fails
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

    // The firstCondBranchOp must be also be an operand
    // forming a cycle with the firstMergeOp; otherwise, the pattern match
    // fails
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

    // At this point we have firstMergeOp and firstCondBranchOp which
    // constitute the first Repeat sturcture. It should feed a second Repeat
    // structure otherwise the pattern match fails Check if secondMergeOp also
    // has a Branch forming a cycle
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

    // The secondCondBranchOp should not have any more successors; otherwise,
    // it is not a Repeat structure
    if (std::distance(secondCondBranchOp->getResults().getUsers().begin(),
                      secondCondBranchOp->getResults().getUsers().end()) != 1)
      return failure();

    // Now, we are sure we have two consecutive Repeats, check the signs of
    // loop conditions. Retrieve the values at the Merges inputs Retrieve the
    // values at the merge inputs
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
      // Insert a NOT at the condition input of the second Branch
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          secondCondBranchOp->getLoc(), condBr2);
      Value newCond = notOp.getResult();
      rewriter.replaceAllUsesWith(condBr2, newCond);

      // Replace all uses coming from the false side of the second Branch with
      // the true side of it
      rewriter.replaceAllUsesWith(secondBranchFalseResult,
                                  secondBranchTrueResult);
      // Adjust the secondTrueIterFlag
      secondTrueIterFlag = true;

      // Retrieve the new value of the condition, in case it is not updated
      condBr2 = secondCondBranchOp.getConditionOperand();

    } else if (!firstTrueIterFlag && secondTrueIterFlag) {
      // Insert a NOT at the condition input of the first Branch
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          firstCondBranchOp->getLoc(), condBr1);
      Value newCond = notOp.getResult();
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

    // The goal now is to replace the two Repeats with a single Repeat, we do
    // so by deleting the first Merge and Branch and adjusting the inputs of
    // the second Merge The new condition is a Merge, calculate its inputs:
    // One input of the Merge will be a constant that should take the value of
    // the condition that feeds a sink (for suppressing) and should be
    // triggered from Source
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
      // This means repeat when the condition is true, so put the constVal at
      // in1 and the additional condition (i.e., condition of the first
      // Repeat) at in0
      muxOperands = {condBr1, constantVal};
    } else {
      assert(!firstTrueIterFlag && !firstTrueIterFlag);
      // This means repeat when the condition is false, so put the constVal at
      // in0 and the additional condition (i.e., the condition of the first
      // Repeat) at in1
      muxOperands = {constantVal, condBr1};
    }
    rewriter.setInsertionPoint(secondCondBranchOp);
    handshake::MuxOp mux = rewriter.create<handshake::MuxOp>(
        secondCondBranchOp->getLoc(), condBr2, muxOperands);
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

    llvm::errs() << "\t***Rules C: shorten-merge-repeat-pairs!***\n";

    return success();
  }
};

// Rules D
// Breaks a Branch that has both true and false successors into two
// Suppresses.
struct ConstructSuppresses
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    // If this Branch does not have users both in the true and false sides,
    // the pattern match fails
    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();
    if (branchTrueResult.getUsers().empty() ||
        branchFalseResult.getUsers().empty())
      return failure();

    // Create a new Branch and let its true side replace the true side of the
    // old Branch
    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = condBranchOp.getConditionOperand();

    ValueRange branchOperands = {condOperand, dataOperand};
    rewriter.setInsertionPoint(condBranchOp);
    handshake::ConditionalBranchOp newBranch =
        rewriter.create<handshake::ConditionalBranchOp>(condBranchOp->getLoc(),
                                                        branchOperands);
    Value newBranchTrueResult = newBranch.getTrueResult();
    rewriter.replaceAllUsesWith(branchTrueResult, newBranchTrueResult);

    llvm::errs() << "\t***Rules D: break-branches!***\n";

    return success();
  }
};

// Rules D
// If a Branch has one successor in the true side, reverse it to be really a
// Suppress
struct FixBranchesToSuppresses
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
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
    handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
        condBranchOp->getLoc(), condBranchOp.getConditionOperand());
    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = notOp.getResult();

    ValueRange branchOperands = {condOperand, dataOperand};
    rewriter.setInsertionPoint(condBranchOp);
    handshake::ConditionalBranchOp newBranch =
        rewriter.create<handshake::ConditionalBranchOp>(condBranchOp->getLoc(),
                                                        branchOperands);
    Value newBranchFalseResult = newBranch.getFalseResult();
    rewriter.replaceAllUsesWith(branchTrueResult, newBranchFalseResult);

    // Commented it out because now I rely on RemoveUselessBranches to erase all
    // such Branches
    // rewriter.eraseOp(condBranchOp);

    llvm::errs() << "\t***Rules D: fix-branches-for-suppresses!***\n";

    return success();
  }
};

// Rules D
// If a Suppress has two or more successors, feed each successor by a separate
// Suppress
struct DistributeSuppresses
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    // If this Branch has any users in the true side, then it is not a
    // suppress, so the pattern match fails. It also fails if it has no users
    // on the false side
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

    llvm::errs() << "\t***Rules D: distribute-suppresses!***\n";

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
    patterns
        .add<EraseUnconditionalBranches, EraseSingleInputMerges,
             EraseSingleInputMuxes, EraseSingleInputControlMerges,
             DowngradeIndexlessControlMerge, RemoveDoubleSinkBranches,
             RemoveMuxFloatingLoop, RemoveMergeFloatingLoop,
             ConstructSuppresses, FixBranchesToSuppresses, DistributeSuppresses,
             ExtractIfThenElseCondition, ExtractLoopCondition,
             RemoveBranchMergeIfThenElse, RemoveBranchMuxIfThenElse,
             RemoveMergeBranchLoop, RemoveMuxBranchLoop>(ctx);

    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

// (1) Test gcd and spmv and sobel, (2) Fix bugs of Shorten, (3) Implement
// Distribute Repeat->Fork, (4) See if you could suggest and explain which of
// rules are enough for which benchmark OR apply only one class at a time and
// collect results..

// ShortenSuppressPairs, ShortenMuxRepeatPairs, ShortenMergeRepeatPairs

// TODO: (1) Test the shorten rules, (2) Implement and test the rule of
// distributing Repeats?, (3) Run full benchmarks
// Might need TODO a rewriter
// that erases any 2 NOTs in a sequence!!!!

// VIP TODO: IN SHORTENING THE REPEAT, MAYBE IT IS WISER TO PUT IT IN THE LOOP
// EXIT BLOCK OF THE OUTER LOOP TO TRICK THE BUFFERS???

// TODO: FInish extract and remove loops and test, check the
// situation of SHorten, add distribute of Repeat? (Rethink when is it needed??)

// TODO: MAKE THE REMOVERULES consistent with the idea of extracting Suppresses
// out of Branches first... Then, implement DistributeRepeats??? Then collect
// numbers

// EraseUnconditionalBranches, EraseSingleInputMerges,
//              EraseSingleInputMuxes, EraseSingleInputControlMerges,
//              DowngradeIndexlessControlMerge, BreakBranches,
//              ExtractIfThenElseMuxCondition, ExtractLoopMuxCondition,
//              RemoveBranchMergeIfThenElse, RemoveBranchMuxIfThenElse,
//              RemoveMergeBranchLoop, RemoveMuxBranchLoop,
//              ShortenSuppressPairs, ShortenMuxRepeatPairs,
//              ShortenMergeRepeatPairs

// WORTH THINKING::: The whole point of the local optimization rules is to make
// the rules of the remove redundant datapath and the shorten datapath
// applicable.. In particular the Repeat distribution over fork makes both
// classes applicable, whereas the Branch-> Fork distibution makes only the
// shorten datapath applicable. Now, due to my current way of implementing the
// remove redundant datapath, we do not need to implement the Repeat->Fork
// distribution in the need future...
// BUT, WE need the rule that breaks one Branch into 2 suppresses...

// TODO: (1) DEbug Sobel, (2) Test the datapath rules: Correctness and
// effectiveness, (3) IMplement the SUPP->SUPP and REP->REP, (4) TODO anything
// for the Supp rule that breaks one Branch into 2??, (5) Do you want to do
// anything for forks (in particular the exchange fork supp rules...), (6) START
// THINKING OF THE BUFFERS PROBLEM!!!

// TODO: (Maybe after the BUffers): Change all rules to start applying (1)
// canonicalization, (2) the rules of local changes, (3) extracting Mux
// conditions, (4) Eliminating redundant datapaths, (5) Shortening datapaths
// Then, add 1 function to the materialization pass (maybe) to
// undo the effect of the local optimization rules by combine Branches AND
// Muxes/Merges........ ALSO, in that additional pass, combine common Merges of
// INIT
std::unique_ptr<dynamatic::DynamaticPass> dynamatic::rewriteHandshakeTerms() {
  return std::make_unique<HandshakeRewriteTermsPass>();
}
