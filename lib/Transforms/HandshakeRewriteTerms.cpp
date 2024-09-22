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

// Removes Conditional Branch and Control Merge operation pairs if both the
// inputs of the Control Merge are outputs of the Conditional Branch. The
// results of the Merge are replaced with the data operand and condition
// operands of the Conditional Branch.
struct RemoveBranchCMergeIfThenElse
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

    // Replace the Cmerge data output with the Branch data input
    Value branchData = condBranchOp.getDataOperand();
    Value branchCondition = condBranchOp.getConditionOperand();

    Value cmergeOutput = cmergeOp.getResult();
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

    // Replace all uses of the branchOuterResult with
    // the cmergeOuterOperand
    rewriter.replaceAllUsesWith(cmergeOutput, branchData);
    rewriter.replaceAllUsesWith(index, cond);

    // Delete the Cmerge
    rewriter.eraseOp(cmergeOp);

    // If the only user of the condBranchOp is the cmerge, delete it
    if (std::distance(trueResUsers.begin(), trueResUsers.end()) == 0 &&
        std::distance(trueResUsers.begin(), trueResUsers.end()) == 0)
      rewriter.eraseOp(condBranchOp);

    llvm::errs()
        << "\t***Completed the remove-branch-cmerge-if-then-else!***\n";
    return success();
  }
};

// Removes Control Merge and Branch operation pairs there exits a loop between
// the Control Merge and the Branch. The index result of the Control Merge is
// derived from a merge operation whose operands are case dependent.
struct RemoveCMergeBranchLoop
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
    Value cmergeOuterOperand = cmergeDataOperands[cmergeOuterInputIdx];
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

    // Replace all uses of the branchOuterResult with
    // the cmergeOuterOperand
    rewriter.replaceAllUsesWith(branchOuterResult, cmergeOuterOperand);

    // Replace all uses of the cmerge index with an INIT (initially with the
    // condition of the Branch then add logic for negation)
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

    // If the only user of the Cmerge output is the condBranchOp, delete both
    // of them
    if (std::distance(cmergeUsers.begin(), cmergeUsers.end()) == 1) {
      rewriter.replaceAllUsesWith(condBranchOp.getDataOperand(),
                                  cmergeOuterOperand);
      rewriter.eraseOp(cmergeOp);
      rewriter.eraseOp(condBranchOp);
    }

    llvm::errs() << "\t***Completed the remove-cmerge-branch-loop!***\n";
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
                 DowngradeIndexlessControlMerge, RemoveBranchCMergeIfThenElse,
                 RemoveCMergeBranchLoop>(ctx);

    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::rewriteHandshakeTerms() {
  return std::make_unique<HandshakeRewriteTermsPass>();
}
