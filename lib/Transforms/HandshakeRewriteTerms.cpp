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

    // Retrieve the values at the mux inputs
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

// Rules C
// Replaces a pair of consecutive Suppress operations with a
// a single suppress operation with a mux at its condition input.
struct ShortenSuppressPairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(handshake::ConditionalBranchOp firstCondBranchOp,
                  PatternRewriter &rewriter) const override {
    // Consider only Branches that either has trueSuccs or falseSuccs but not
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

    // For the shortening to work, the two branches should have their successor
    // in the same direction (either true or false); otherwise, we need to
    // enforce it by negating.. When they are not consistent, we will force both
    // to have their succs in the false side and sink in true side (like a
    // typical suppress)
    Value condBr1 = firstCondBranchOp.getConditionOperand();
    Value condBr2 = secondCondBranchOp.getConditionOperand();
    if (firstTrueSuccOnlyFlag && secondFalseSuccOnlyFlag) {
      // Insert a NOT at the condition input of the first Branch
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          firstCondBranchOp->getLoc(), condBr1);
      Value newCond = notOp.getResult();
      rewriter.replaceAllUsesWith(condBr1, newCond);

      // Replace all uses coming from the true side of the first Branch with the
      // false side of it
      rewriter.replaceAllUsesWith(firstTrueResult, firstFalseResult);
      // Adjust the firstTrueSuccOnlyFlag and firstFalseSuccOnlyFlag
      firstTrueSuccOnlyFlag = false;
      firstFalseSuccOnlyFlag = true;

      // Retrieve the new value of the condition, in case it is not updated
      condBr1 = firstCondBranchOp.getConditionOperand();
    } else {
      assert(firstFalseSuccOnlyFlag && secondTrueSuccOnlyFlag);
      // Insert a NOT at the condition input of the second Branch
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          secondCondBranchOp->getLoc(), condBr2);
      Value newCond = notOp.getResult();
      rewriter.replaceAllUsesWith(condBr2, newCond);

      // Replace all uses coming from the true side of the first Branch with the
      // false side of it
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
      // This means suppress when the condition is false, so put the constVal at
      // in0 and the additional condition at in1
      muxOperands = {constantVal, condBr2};
    } else {
      assert(firstFalseSuccOnlyFlag && secondFalseSuccOnlyFlag);
      // This means suppress when the condition is true, so put the constVal at
      // in1 and the additional condition at in0
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

    // If the mux is not driven by a Merge (i.e., INIT), the pattern match fails
    if (!isa_and_nonnull<handshake::MergeOp>(
            firstMuxOp.getSelectOperand().getDefiningOp()))
      return failure();

    // One user must be a Branch; otherwise, the pattern match fails
    bool firstFoundCondBranch = false;
    handshake::ConditionalBranchOp firstCondBranchOp;
    // One user must be another Mux belonging to a second Repeat; otherwise, the
    // pattern match fails
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

    // The firstCondBranchOp should not have any more successors; otherwise, it
    // is not a Repeat structure
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

    // If the mux is not driven by a Merge (i.e., INIT), the pattern match fails
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

    // The secondCondBranchOp should not have any more successors; otherwise, it
    // is not a Repeat structure
    if (std::distance(secondCondBranchOp->getResults().getUsers().begin(),
                      secondCondBranchOp->getResults().getUsers().end()) != 1)
      return failure();

    // Now, we are sure we have two consecutive Repeats, check the signs of loop
    // conditions.
    // Retrieve the values at the Muxes inputs
    // Retrieve the values at the mux inputs
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
      // in1 and the additional condition (i.e., condition of the first Repeat)
      // at in0
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

    // Correct the select of the second Mux; at this point, we are sure it comes
    // from a Merge (INIT), so retrieve it
    assert(isa_and_nonnull<handshake::MergeOp>(
        secondMuxOp.getSelectOperand().getDefiningOp()));
    handshake::MergeOp initOp = cast<handshake::MergeOp>(
        secondMuxOp.getSelectOperand().getDefiningOp());
    // The convention used in the ExtractLoopMuxCondition rewrite puts the loop
    // condition at in0 of the Merge
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

    // TODO: Erase the first INIT as well
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
    // (1) Get the users of the Merge. If they are not exactly two, the pattern
    // match fails
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
    // forming a cycle with the firstMergeOp; otherwise, the pattern match fails
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

    // The firstCondBranchOp should not have any more successors; otherwise, it
    // is not a Repeat structure
    if (std::distance(firstCondBranchOp->getResults().getUsers().begin(),
                      firstCondBranchOp->getResults().getUsers().end()) != 1)
      return failure();

    // At this point we have firstMergeOp and firstCondBranchOp which constitute
    // the first Repeat sturcture. It should feed a second Repeat structure
    // otherwise the pattern match fails
    // Check if secondMergeOp also has a Branch forming a cycle
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

    // The secondCondBranchOp should not have any more successors; otherwise, it
    // is not a Repeat structure
    if (std::distance(secondCondBranchOp->getResults().getUsers().begin(),
                      secondCondBranchOp->getResults().getUsers().end()) != 1)
      return failure();

    // Now, we are sure we have two consecutive Repeats, check the signs of loop
    // conditions.
    // Retrieve the values at the Merges inputs
    // Retrieve the values at the merge inputs
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
    // so by deleting the first Merge and Branch and adjusting the inputs of the
    // second Merge The new condition is a Merge, calculate its inputs: One
    // input of the Merge will be a constant that should take the value of the
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
      // in1 and the additional condition (i.e., condition of the first Repeat)
      // at in0
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
                 RemoveMuxBranchLoop, ShortenSuppressPairs,
                 ShortenMuxRepeatPairs, ShortenMergeRepeatPairs>(ctx);

    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::rewriteHandshakeTerms() {
  return std::make_unique<HandshakeRewriteTermsPass>();
}
