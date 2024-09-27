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
    llvm::errs() << "\t***Removing unconditional Branch!!***\n";
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

// TODO: Extend it to assume that the two Branches can be two different units
// that have the same condition!!!
// Rules A
// Removes Conditional Branch and Merge operation pairs if both the
// inputs of the Merge are outputs of the Conditional Branch. The
// results of the Merge are replaced with the data operand of the Conditional
// Branch.
struct RemoveBranchMergeIfThenElse
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    // Make an extra check that the Branch has users both for the true and false
    // successors
    auto trueResUsers = condBranchOp.getTrueResult().getUsers();
    auto falseResUsers = condBranchOp.getFalseResult().getUsers();
    if (trueResUsers.empty() || falseResUsers.empty())
      return failure();

    // If there is not a single Merge that is both in the trueResUsers and
    // the falseResUsers, the pattern match fails
    bool foundMerge = false;
    handshake::MergeOp mergeOp;
    for (auto trueSucc : trueResUsers) {
      for (auto falseSucc : falseResUsers) {
        if (trueSucc == falseSucc &&
            isa_and_nonnull<handshake::MergeOp>(trueSucc)) {
          foundMerge = true;
          mergeOp = cast<handshake::MergeOp>(trueSucc);
          break;
        }
      }
    }
    if (!foundMerge)
      return failure();

    // Doublecheck that the Merge has 2 inputs; otherwise, the pattern match
    // fails
    if (mergeOp->getNumOperands() != 2)
      return failure();

    // The two inputs of the Merge should be the condBranchOp; otherwise the
    // pattern match fails
    auto mergeOpOperands = mergeOp.getOperands();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(
            mergeOpOperands[0].getDefiningOp()) ||
        !isa_and_nonnull<handshake::ConditionalBranchOp>(
            mergeOpOperands[1].getDefiningOp()))
      return failure();

    handshake::ConditionalBranchOp mergeOperand1 =
        cast<handshake::ConditionalBranchOp>(
            mergeOpOperands[0].getDefiningOp());
    handshake::ConditionalBranchOp mergeOperand2 =
        cast<handshake::ConditionalBranchOp>(
            mergeOpOperands[1].getDefiningOp());
    if (mergeOperand1 != condBranchOp || mergeOperand2 != condBranchOp)
      return failure();

    Value branchData = condBranchOp.getDataOperand();
    Value mergeOutput = mergeOp.getResult();

    // Replace all uses of the branchOuterResult with
    // the cmergeOuterOperand
    rewriter.replaceAllUsesWith(mergeOutput, branchData);
    // Delete the merge
    rewriter.eraseOp(mergeOp);

    // If the only user of the condBranchOp is the merge, delete it
    if (std::distance(trueResUsers.begin(), trueResUsers.end()) == 1 &&
        std::distance(trueResUsers.begin(), trueResUsers.end()) == 1)
      rewriter.eraseOp(condBranchOp);

    llvm::errs() << "\t***Completed the remove-branch-merge-if-then-else!***\n";
    return success();
  }
};

// TODO: Extend it to assume that the two Branches can be two different units
// that have the same condition!!!
// Rules A Removes Conditional Branch and Mux
// operation pairs if both the inputs of the Mux are outputs of the Conditional
// Branch. The results of the MeMuxrge are replaced with the data operand.
struct RemoveBranchMuxIfThenElse
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    // Make an extra check that the Branch has users both for the true and false
    // successors
    auto trueResUsers = condBranchOp.getTrueResult().getUsers();
    auto falseResUsers = condBranchOp.getFalseResult().getUsers();
    if (trueResUsers.empty() || falseResUsers.empty())
      return failure();

    // If there is not a single Mux that is both in the trueResUsers and
    // the falseResUsers, the pattern match fails
    bool foundMux = false;
    handshake::MuxOp muxOp;
    for (auto trueSucc : trueResUsers) {
      for (auto falseSucc : falseResUsers) {
        if (trueSucc == falseSucc &&
            isa_and_nonnull<handshake::MuxOp>(trueSucc)) {
          foundMux = true;
          muxOp = cast<handshake::MuxOp>(trueSucc);
          break;
        }
      }
    }
    if (!foundMux)
      return failure();

    // Doublecheck that the Mux has 2 inputs; otherwise, the pattern match
    // fails
    if (muxOp->getNumOperands() != 2)
      return failure();

    // The two inputs of the Mux should be the condBranchOp; otherwise the
    // pattern match fails
    auto muxOpOperands = muxOp.getOperands();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(
            muxOpOperands[0].getDefiningOp()) ||
        !isa_and_nonnull<handshake::ConditionalBranchOp>(
            muxOpOperands[1].getDefiningOp()))
      return failure();

    handshake::ConditionalBranchOp muxOperand1 =
        cast<handshake::ConditionalBranchOp>(muxOpOperands[0].getDefiningOp());
    handshake::ConditionalBranchOp muxOperand2 =
        cast<handshake::ConditionalBranchOp>(muxOpOperands[1].getDefiningOp());
    if (muxOperand1 != condBranchOp || muxOperand2 != condBranchOp)
      return failure();

    Value branchData = condBranchOp.getDataOperand();
    Value muxOutput = muxOp.getResult();

    // Replace all uses of the branchOuterResult with
    // the cmergeOuterOperand
    rewriter.replaceAllUsesWith(muxOutput, branchData);
    // Delete the merge
    rewriter.eraseOp(muxOp);

    // If the only user of the condBranchOp is the mux, delete it
    if (std::distance(trueResUsers.begin(), trueResUsers.end()) == 1 &&
        std::distance(trueResUsers.begin(), trueResUsers.end()) == 1)
      rewriter.eraseOp(condBranchOp);

    llvm::errs() << "\t***Completed the remove-branch-mux-if-then-else!***\n";
    return success();
  }
};

// TODO: Extend it to assume that the two Branches can be two different units
// that have the same condition!!!
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

    llvm::errs() << "\t\t(1) Found a Merge-Branch loop";

    // One of the branches in the set of Branches must be also be an operand
    // forming a cycle with the merge; otherwise, the pattern match fails
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
    mergeOuterInputIdx = (mergeCycleInputIdx == 0) ? 1 : 0;

    llvm::errs() << "\t\t(2) Found a Merge-Branch loop";

    // Retrieve the values at the merge inputs
    OperandRange mergeDataOperands = mergeOp.getDataOperands();
    Value mergeOuterOperand = mergeDataOperands[mergeOuterInputIdx];
    Value mergeInnerOperand = mergeDataOperands[mergeCycleInputIdx];

    // Identify the output of the Branch going outside of the loop
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

    llvm::errs() << "\t\t(3) Found a Merge-Branch loop";

    // If the output of the Branch going outside of the loop is not used by
    // anyone and the merge has users other than the Branch, then we do not need
    // to do anything
    if (std::distance(mergeUsers.begin(), mergeUsers.end()) > 1 &&
        branchOuterResult.getUsers().empty()) {
      llvm::errs() << "\t\tExited because branch has no outer!!\n";
      return failure();
    }

    llvm::errs() << "\t\t(4) Found a Merge-Branch loop";

    // Replace all uses of the branchOuterResult with
    // the mergeOuterOperand
    rewriter.replaceAllUsesWith(branchOuterResult, mergeOuterOperand);

    // If the only user of the merge output is the condBranchOp AND the only
    // user of the condBranchOp's iterator output is the merge, delete both of
    // them
    if (std::distance(mergeUsers.begin(), mergeUsers.end()) == 1 &&
        ((!isTrueOutputOuter &&
          std::distance(branchTrueResult.getUsers().begin(),
                        branchTrueResult.getUsers().end()) == 1) ||
         (isTrueOutputOuter &&
          std::distance(branchFalseResult.getUsers().begin(),
                        branchFalseResult.getUsers().end()) == 1))) {
      rewriter.replaceAllUsesWith(condBranchOp.getDataOperand(),
                                  mergeOuterOperand);
      rewriter.eraseOp(mergeOp);
      rewriter.eraseOp(condBranchOp);
    }

    llvm::errs() << "\t***Completed the remove-merge-branch-loop!***\n";

    return success();
  }
};

// TODO: Extend it to assume that the two Branches can be two different units
// that have the same condition!!!
// Rules A
// Removes Mux and Branch operation pairs if there exits a loop between
// the Mux and the Branch.
struct RemoveMuxBranchLoop : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {

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
    // forming a cycle with the mux; otherwise, the pattern match fails
    bool foundCycle = false;
    int operIdx = 0;
    int muxOuterInputIdx = 0;
    int muxCycleInputIdx = 0;
    handshake::ConditionalBranchOp condBranchOp;
    for (auto muxOperand : muxOp->getOperands()) {
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
    muxOuterInputIdx = (muxCycleInputIdx == 0) ? 1 : 0;

    // Retrieve the values at the merge inputs
    OperandRange muxDataOperands = muxOp.getDataOperands();
    Value muxOuterOperand = muxDataOperands[muxOuterInputIdx];
    Value muxInnerOperand = muxDataOperands[muxCycleInputIdx];

    // Identify the output of the Branch going outside of the loop
    bool isTrueOutputOuter = false;
    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();
    Value branchOuterResult;
    if (branchTrueResult == muxInnerOperand)
      branchOuterResult = branchFalseResult;
    else if (branchFalseResult == muxInnerOperand) {
      branchOuterResult = branchTrueResult;
      isTrueOutputOuter = true;
    } else
      return failure();

    // If the output of the Branch going outside of the loop is not used by
    // anyone and the merge has users other than the Branch, then we do not need
    // to do anything
    if (std::distance(muxUsers.begin(), muxUsers.end()) > 1 &&
        branchOuterResult.getUsers().empty())
      return failure();

    // Replace all uses of the branchOuterResult with
    // the mergeOuterOperand
    rewriter.replaceAllUsesWith(branchOuterResult, muxOuterOperand);

    // If the only user of the merge output is the condBranchOp AND the only
    // user of the condBranchOp's iterator output is the merge, delete both of
    // them
    if (std::distance(muxUsers.begin(), muxUsers.end()) == 1 &&
        ((!isTrueOutputOuter &&
          std::distance(branchTrueResult.getUsers().begin(),
                        branchTrueResult.getUsers().end()) == 1) ||
         (isTrueOutputOuter &&
          std::distance(branchFalseResult.getUsers().begin(),
                        branchFalseResult.getUsers().end()) == 1))) {
      rewriter.replaceAllUsesWith(condBranchOp.getDataOperand(),
                                  muxOuterOperand);
      rewriter.eraseOp(muxOp);
      rewriter.eraseOp(condBranchOp);
    }

    llvm::errs() << "\t***Completed the remove-mux-branch-loop!***\n";

    return success();
  }
};

// Rules B
// Extract the index result of the Control Merge in a loop structure.
struct ExtractLoopMuxCondition
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
    handshake::ConditionalBranchOp condBranchOp;
    for (auto cmergeUser : cmergeUsers) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(cmergeUser)) {
        foundCondBranch = true;
        condBranchOp = cast<handshake::ConditionalBranchOp>(cmergeUser);
        break;
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
    for (auto cmergeOperand : cmergeOp->getOperands()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(
              cmergeOperand.getDefiningOp()))
        if (cast<handshake::ConditionalBranchOp>(
                cmergeOperand.getDefiningOp()) == condBranchOp) {
          foundCycle = true;
          cmergeCycleInputIdx = operIdx;
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

    // Identify the output of the Branch going outside of the loop
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

    llvm::errs() << "\t***Completed the extract-loop-mux-condition!***\n";
    return success();
  }
};

// Rules B
// Extract the index result of the Control Merge in an if-then-else structure.
struct ExtractIfThenElseMuxCondition
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    // Make an extra check that the Branch has users both for the true and false
    // successors
    auto trueResUsers = condBranchOp.getTrueResult().getUsers();
    auto falseResUsers = condBranchOp.getFalseResult().getUsers();
    if (trueResUsers.empty() || falseResUsers.empty())
      return failure();

    // If there is not a single Cmerge that is both in the trueResUsers and
    // the falseResUsers, the pattern match fails
    bool foundCmerge = false;
    handshake::ControlMergeOp cmergeOp;
    for (auto trueSucc : trueResUsers) {
      for (auto falseSucc : falseResUsers) {
        if (trueSucc == falseSucc &&
            isa_and_nonnull<handshake::ControlMergeOp>(trueSucc)) {
          foundCmerge = true;
          cmergeOp = cast<handshake::ControlMergeOp>(trueSucc);
          break;
        }
      }
    }
    if (!foundCmerge)
      return failure();

    // Doublecheck that the CMerge has 2 inputs; otherwise, the pattern match
    // fails
    if (cmergeOp->getNumOperands() != 2)
      return failure();

    // The two inputs of the Cmerge should be the condBranchOp; otherwise the
    // pattern match fails
    auto cmergeOpOperands = cmergeOp.getOperands();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(
            cmergeOpOperands[0].getDefiningOp()) ||
        !isa_and_nonnull<handshake::ConditionalBranchOp>(
            cmergeOpOperands[1].getDefiningOp()))
      return failure();
    handshake::ConditionalBranchOp cmergeOperand1 =
        cast<handshake::ConditionalBranchOp>(
            cmergeOpOperands[0].getDefiningOp());
    handshake::ConditionalBranchOp cmergeOperand2 =
        cast<handshake::ConditionalBranchOp>(
            cmergeOpOperands[1].getDefiningOp());
    if (cmergeOperand1 != condBranchOp || cmergeOperand2 != condBranchOp)
      return failure();

    Value branchCondition = condBranchOp.getConditionOperand();
    Value index = cmergeOp.getIndex();

    // Check if we need to negate the condition before feeding it to the index
    // output of the cmerge
    bool needNot = (condBranchOp.getTrueResult() == cmergeOpOperands[0] &&
                    condBranchOp.getFalseResult() == cmergeOpOperands[1]);
    Value cond;
    if (needNot) {
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          cmergeOp->getLoc(), branchCondition);
      cond = notOp.getResult();
    } else {
      cond = branchCondition;
    }

    // Replace the Cmerge index output with the branch condition
    rewriter.replaceAllUsesWith(index, cond);

    llvm::errs()
        << "\t***Completed the extract-if-then-else-mux-condition!***\n";
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
    patterns.add<EraseUnconditionalBranches, EraseSingleInputMerges,
                 EraseSingleInputMuxes, EraseSingleInputControlMerges,
                 DowngradeIndexlessControlMerge, ExtractIfThenElseMuxCondition,
                 ExtractLoopMuxCondition, RemoveBranchMergeIfThenElse,
                 RemoveBranchMuxIfThenElse, RemoveMergeBranchLoop,
                 RemoveMuxBranchLoop>(ctx);

    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::rewriteHandshakeTerms() {
  return std::make_unique<HandshakeRewriteTermsPass>();
}
