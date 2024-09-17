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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <vector>

using namespace mlir;
using namespace dynamatic;

namespace {

// Removes Conditional Branch and Control Merge operation pairs if both the
// inputs of the Control Merge are outputs of the Conditional Branch. The
// results of the Merge are replaced with the data operand and condition
// operands of the Conditional Branch.
struct RemoveBranchCMergePairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    Operation *useOwner1 = nullptr;
    Operation *useOwner2 = nullptr;

    auto trueResUsers = condBranchOp.getTrueResult().getUsers();
    if (trueResUsers.empty())
      return failure();
    useOwner1 = *trueResUsers.begin();
    auto falseResUsers = condBranchOp.getFalseResult().getUsers();
    if (falseResUsers.empty())
      return failure();
    useOwner2 = *falseResUsers.begin();

    if (!isa_and_nonnull<handshake::ControlMergeOp>(useOwner1) ||
        useOwner1 != useOwner2)
      return failure();

    handshake::ControlMergeOp cMergeOp =
        cast<handshake::ControlMergeOp>(useOwner1);
    if (cMergeOp->getNumOperands() != 2)
      return failure();

    Value dataOperand = condBranchOp.getDataOperand();
    Value conditionBr = condBranchOp.getConditionOperand();

    rewriter.replaceOp(cMergeOp, {dataOperand, conditionBr});
    rewriter.eraseOp(condBranchOp);
    return success();
  }
};

// Removes Control Merge and Branch operation pairs there exits a loop between
// the Control Merge and the Branch. The index result of the Control Merge is
// derived from a merge operation whose operands are case dependent.
struct RemoveCMergeBranchLoopPairs
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
    // Create a new ConstantOp in the same block as that of the branch forming
    // the cycle
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

    // If the only user of the Cmerge output is the condBranchOp, delete both of
    // them
    if (std::distance(cmergeUsers.begin(), cmergeUsers.end()) == 1) {
      rewriter.replaceAllUsesWith(condBranchOp.getDataOperand(),
                                  cmergeOuterOperand);
      rewriter.eraseOp(cmergeOp);
      rewriter.eraseOp(condBranchOp);
    }

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
    patterns.add<RemoveBranchCMergePairs, RemoveCMergeBranchLoopPairs>(ctx);

    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::rewriteHandshakeTerms() {
  return std::make_unique<HandshakeRewriteTermsPass>();
}
