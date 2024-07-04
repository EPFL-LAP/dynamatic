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
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include <vector>

using namespace mlir;
using namespace dynamatic;

namespace {

struct RemoveBranchMuxPairs
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

    if (!isa_and_nonnull<handshake::MuxOp>(useOwner1) || useOwner1 != useOwner2)
      return failure();
    handshake::MuxOp muxOp = cast<handshake::MuxOp>(useOwner1);
    Value conditionBr = condBranchOp.getConditionOperand();
    Value selectMux = muxOp.getSelectOperand();

    if (conditionBr.getDefiningOp() != selectMux.getDefiningOp() ||
        !isa_and_nonnull<handshake::ForkOp>(conditionBr.getDefiningOp()))
      return failure();
    Value dataOperand = condBranchOp.getDataOperand();
    rewriter.create<handshake::SinkOp>(muxOp->getLoc(), selectMux);
    rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), conditionBr);
    rewriter.replaceOp(muxOp, dataOperand);
    rewriter.eraseOp(condBranchOp);
    return success();
  }
};

struct RemoveBranchMergePairs
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

    if (!isa_and_nonnull<handshake::MergeOp>(useOwner1) ||
        useOwner1 != useOwner2)
      return failure();
    handshake::MergeOp mergeOp = cast<handshake::MergeOp>(useOwner1);
    if (mergeOp->getNumOperands() != 2)
      return failure();
    Value conditionBr = condBranchOp.getConditionOperand();
    Value dataOperand = condBranchOp.getDataOperand();
    rewriter.replaceOp(mergeOp, dataOperand);
    rewriter.setInsertionPoint(condBranchOp);
    rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), conditionBr);
    rewriter.eraseOp(condBranchOp);
    return success();
  }
};

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

struct RemoveMuxBranchLoopPairs : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    Operation *useOwner = nullptr;
    auto it = muxOp->getUsers().begin();
    if (it != muxOp->getUsers().end())
      useOwner = *it;
    if (muxOp.getNumOperands() != 2)
      return failure();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner))
      return failure();
    handshake::ConditionalBranchOp condBranchOp =
        cast<handshake::ConditionalBranchOp>(useOwner);
    OperandRange dataOperands = muxOp.getDataOperands();
    Value first = dataOperands[0];
    Value second = dataOperands[1];
    Value resultMux = muxOp.getResult();
    Value select = muxOp.getSelectOperand();
    Value condition = condBranchOp.getConditionOperand();
    Value branchIn = condBranchOp.getDataOperand();
    Value trueResult = condBranchOp.getTrueResult();
    Value falseResult = condBranchOp.getFalseResult();
    ValueRange l = {second, second};
    if (resultMux != branchIn)
      return failure();
    Value srcVal;
    if (trueResult == first || trueResult == second)
      srcVal = falseResult;
    else if (falseResult == first || falseResult == second)
      srcVal = trueResult;
    else
      return failure();
    Value dstVal =
        (trueResult == first || falseResult == first) ? second : first;
    rewriter.setInsertionPoint(muxOp);
    rewriter.create<handshake::SinkOp>(muxOp->getLoc(), select);
    rewriter.replaceOp(muxOp, dstVal);
    rewriter.replaceAllUsesWith(srcVal, dstVal);
    rewriter.setInsertionPoint(condBranchOp);
    rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), condition);
    rewriter.eraseOp(condBranchOp);
    return success();
  }
};

struct RemoveMergeBranchLoopPairs
    : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
    Operation *useOwner = nullptr;
    auto it = mergeOp->getUsers().begin();
    if (it != mergeOp->getUsers().end())
      useOwner = *it;
    if (mergeOp->getNumOperands() != 2)
      return failure();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner))
      return failure();
    handshake::ConditionalBranchOp condBranchOp =
        cast<handshake::ConditionalBranchOp>(useOwner);

    OperandRange dataOperands = mergeOp.getDataOperands();
    Value first = (dataOperands[0]);
    Value second = (dataOperands[1]);
    Value resultMerge = mergeOp.getResult();
    Value condition = condBranchOp.getConditionOperand();
    Value branchIn = condBranchOp.getDataOperand();
    Value trueResult = condBranchOp.getTrueResult();
    Value falseResult = condBranchOp.getFalseResult();

    if (resultMerge != branchIn)
      return failure();

    Value srcVal;
    if (trueResult == first || trueResult == second)
      srcVal = falseResult;
    else if (falseResult == first || falseResult == second)
      srcVal = trueResult;
    else
      return failure();
    Value dstVal =
        (trueResult == first || falseResult == first) ? second : first;

    rewriter.replaceOp(mergeOp, dstVal);
    rewriter.replaceAllUsesWith(srcVal, dstVal);
    rewriter.setInsertionPoint(condBranchOp);
    rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), condition);
    rewriter.eraseOp(condBranchOp);
    return success();
  }
};

struct RemoveCMergeBranchLoopPairs
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    Block *parentBlock = cmergeOp->getBlock();
    MutableArrayRef<BlockArgument> l = parentBlock->getArguments();
    if (l.empty())
      return failure();
    BlockArgument start = l.back();
    if (!isa<NoneType>(start.getType()))
      return failure();
    Operation *useOwner = nullptr;
    auto it = cmergeOp->getUsers().begin();
    if (it != cmergeOp->getUsers().end())
      useOwner = *it;
    if (cmergeOp->getNumOperands() != 2)
      return failure();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner))
      return failure();
    handshake::ConditionalBranchOp condBranchOp =
        cast<handshake::ConditionalBranchOp>(useOwner);

    OperandRange dataOperands = cmergeOp.getDataOperands();
    Value first = dataOperands[0];
    Value second = dataOperands[1];
    Value resultCMerge = cmergeOp.getResult();
    Value index = cmergeOp.getIndex();
    Value condition = condBranchOp.getConditionOperand();
    Value branchIn = condBranchOp.getDataOperand();
    Value trueResult = condBranchOp.getTrueResult();
    Value falseResult = condBranchOp.getFalseResult();

    if (resultCMerge != branchIn)
      return failure();
    Type constantType = rewriter.getIntegerType(1);
    Value zero = rewriter.create<handshake::ConstantOp>(
        cmergeOp->getLoc(), constantType,
        rewriter.getIntegerAttr(constantType, 0), start);
    Value one = rewriter.create<handshake::ConstantOp>(
        cmergeOp->getLoc(), constantType,
        rewriter.getIntegerAttr(constantType, 1), start);
    handshake::NotOp notOp =
        rewriter.create<handshake::NotOp>(cmergeOp->getLoc(), condition);

    Value srcVal;
    if (trueResult == first || trueResult == second)
      srcVal = falseResult;
    else if (falseResult == first || falseResult == second)
      srcVal = trueResult;
    else
      return failure();
    Value dstVal =
        (trueResult == first || falseResult == first) ? second : first;
    Value valueOfConstant =
        (trueResult == first || falseResult == first) ? one : zero;
    Value inputToMerge = (trueResult == first || falseResult == second)
                             ? notOp.getResult()
                             : condition;

    rewriter.replaceAllUsesWith(srcVal, dstVal);
    rewriter.replaceAllUsesWith(resultCMerge, dstVal);
    ValueRange operands = {inputToMerge, valueOfConstant};
    rewriter.setInsertionPointAfter(cmergeOp);
    handshake::MergeOp mergeOp =
        rewriter.create<handshake::MergeOp>(cmergeOp.getLoc(), operands);
    Value mergeResult = mergeOp.getResult();
    rewriter.replaceAllUsesWith(index, mergeResult);
    rewriter.eraseOp(cmergeOp);
    rewriter.eraseOp(condBranchOp);
    return success();
  }
};

struct RemoveConsecutiveForksPairs
    : public OpRewritePattern<handshake::ForkOp> {
  using OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp fork1,
                                PatternRewriter &rewriter) const override {
    Operation *useOwner = nullptr;
    Value input = fork1.getOperand();
    ResultRange::user_range forkUsers = fork1->getUsers();
    int numNewResultsReqd = 0;
    int numResultsFork = fork1.getNumResults();
    handshake::ForkOp forkNext;
    for (auto it = forkUsers.begin(); it != forkUsers.end(); it++) {
      useOwner = *it;
      if (isa_and_nonnull<handshake::ForkOp>(useOwner)) {
        forkNext = cast<handshake::ForkOp>(useOwner);
        numNewResultsReqd += forkNext->getNumResults();
        numResultsFork--;
      }
    }
    if (numNewResultsReqd == 0)
      return failure();
    handshake::ForkOp newFork = rewriter.create<handshake::ForkOp>(
        fork1.getLoc(), input, numResultsFork + numNewResultsReqd);

    ResultRange newForkResults = newFork->getResults();

    int curNew = 0;
    int curOld = 0;

    ResultRange forkResults = fork1.getResults();
    for (auto it = forkUsers.begin(); it != forkUsers.end(); it++) {
      useOwner = *it;
      if (isa_and_nonnull<handshake::ForkOp>(useOwner)) {
        forkNext = cast<handshake::ForkOp>(useOwner);
        ResultRange forkNextResults = forkNext.getResults();
        for (unsigned j{0}; j < forkNext->getNumResults(); j++) {
          rewriter.replaceAllUsesWith(forkNextResults[j],
                                      newForkResults[curNew++]);
        }
        rewriter.eraseOp(forkNext);
      } else
        rewriter.replaceAllUsesWith(forkResults[curOld],
                                    newForkResults[curNew++]);
      curOld++;
    }
    rewriter.eraseOp(fork1);
    return success();
  }
};

struct RemoveSupressForkPairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value trueResult = condBranchOp.getTrueResult();
    Value falseResult = condBranchOp.getFalseResult();
    if (hasRealUses(trueResult))
      return failure();
    Operation *useOwner = nullptr;
    auto op = falseResult.getUsers().begin();
    if (op == falseResult.getUsers().end())
      return failure();
    useOwner = *op;
    if (!isa_and_nonnull<handshake::ForkOp>(useOwner))
      return failure();
    handshake::ForkOp forkOp = cast<handshake::ForkOp>(useOwner);
    int numOfResults = forkOp->getNumResults();
    ResultRange forkResult = forkOp->getResults();

    Value dataOperand = condBranchOp.getDataOperand();
    Value condBr = condBranchOp.getConditionOperand();

    handshake::ForkOp forkForDataop = rewriter.create<handshake::ForkOp>(
        condBranchOp.getLoc(), dataOperand, numOfResults);
    rewriter.setInsertionPointAfter(forkForDataop);

    handshake::ForkOp forkForCondition = rewriter.create<handshake::ForkOp>(
        condBranchOp.getLoc(), condBr, numOfResults);

    ResultRange forkDataResult = forkForDataop->getResults();
    ResultRange forkCondResult = forkForCondition->getResults();

    std::vector<handshake::ConditionalBranchOp> vectorForSuppress(numOfResults);
    std::vector<handshake::SinkOp> vectorForSinks(numOfResults);

    eraseSinkUsers(trueResult, rewriter);
    ValueRange replaceOperands = {dataOperand, dataOperand};
    rewriter.replaceOp(condBranchOp, replaceOperands);

    rewriter.setInsertionPointAfter(forkForCondition);
    for (int i{0}; i < numOfResults; i++) {

      vectorForSuppress[i] = rewriter.create<handshake::ConditionalBranchOp>(
          forkOp.getLoc(), forkCondResult[i], forkDataResult[i]);

      rewriter.setInsertionPointAfter(vectorForSuppress[i]);
      vectorForSinks[i] = rewriter.create<handshake::SinkOp>(
          vectorForSuppress[i].getLoc(), vectorForSuppress[i].getTrueResult());

      rewriter.replaceAllUsesWith(forkResult[i],
                                  vectorForSuppress[i].getFalseResult());

      rewriter.setInsertionPointAfter(vectorForSinks[i]);
    }
    return success();
  }
};

struct RemoveSuppressSuppressPairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value trueResult = condBranchOp.getTrueResult();
    Value falseResult = condBranchOp.getFalseResult();
    if (hasRealUses(trueResult))
      return failure();
    Operation *useOwner = nullptr;
    auto op = falseResult.getUsers().begin();
    if (op == falseResult.getUsers().end())
      return failure();
    useOwner = *op;
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner))
      return failure();

    handshake::ConditionalBranchOp suppress2 =
        cast<handshake::ConditionalBranchOp>(useOwner);
    Value trueResult2 = suppress2.getTrueResult();

    if (hasRealUses(trueResult2))
      return failure();

    Value dataOperand = condBranchOp.getDataOperand();
    Value condBr1 = condBranchOp.getConditionOperand();
    Value condBr2 = suppress2.getConditionOperand();
    Value source = rewriter.create<handshake::SourceOp>(condBranchOp->getLoc());
    int64_t constantValue = 1;
    Type constantType = rewriter.getIntegerType(1);
    Value constantOne = rewriter.create<handshake::ConstantOp>(
        condBranchOp->getLoc(), constantType,
        rewriter.getIntegerAttr(constantType, constantValue), source);
    ValueRange muxOperands = {condBr2, constantOne};
    rewriter.setInsertionPoint(suppress2);
    handshake::MuxOp mux = rewriter.create<handshake::MuxOp>(
        condBranchOp->getLoc(), condBr1, muxOperands);
    Value result = mux.getResult();
    ValueRange operands = {result, dataOperand};
    suppress2->setOperands(operands);
    rewriter.setInsertionPointAfter(suppress2);
    rewriter.create<handshake::SinkOp>(suppress2.getLoc(),
                                       suppress2.getTrueResult());
    eraseSinkUsers(trueResult, rewriter);
    ValueRange replaceOperands = {dataOperand, dataOperand};
    rewriter.replaceOp(condBranchOp, replaceOperands);
    return success();
  }
};

struct BranchToSupressForkPairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    Value dataOperand = condBranchOp.getDataOperand();
    Value condition = condBranchOp.getConditionOperand();
    Value falseResult = condBranchOp.getFalseResult();
    Value trueResult = condBranchOp.getTrueResult();
    if (!hasRealUses(falseResult) || !hasRealUses(trueResult))
      return failure();
    handshake::ForkOp forkData = rewriter.create<handshake::ForkOp>(
        condBranchOp->getLoc(), dataOperand, 2);
    rewriter.setInsertionPointAfter(forkData);
    handshake::ForkOp forkCond = rewriter.create<handshake::ForkOp>(
        condBranchOp->getLoc(), condition, 2);
    rewriter.setInsertionPointAfter(forkCond);
    handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
        condBranchOp->getLoc(), forkCond->getResults()[1]);
    rewriter.setInsertionPointAfter(notOp);
    handshake::ConditionalBranchOp suppress1 =
        rewriter.create<handshake::ConditionalBranchOp>(
            condBranchOp->getLoc(), forkCond->getResults()[0],
            forkData.getResults()[0]);
    rewriter.setInsertionPointAfter(suppress1);
    handshake::SinkOp a = rewriter.create<handshake::SinkOp>(
        condBranchOp->getLoc(), suppress1.getTrueResult());
    rewriter.setInsertionPointAfter(a);
    handshake::ConditionalBranchOp suppress2 =
        rewriter.create<handshake::ConditionalBranchOp>(
            condBranchOp->getLoc(), notOp.getResult(),
            forkData->getResults()[1]);
    rewriter.setInsertionPointAfter(suppress2);
    rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(),
                                       suppress2.getTrueResult());
    ValueRange results = {suppress1.getFalseResult(),
                          suppress2.getFalseResult()};
    rewriter.replaceOp(condBranchOp, results);
    return success();
  }
};

struct RemoveForkSupressPairsMux : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    ValueRange muxInputs = muxOp.getDataOperands();
    Value select = muxOp.getSelectOperand();
    Operation *useOwner1 = muxInputs[0].getDefiningOp();
    Operation *useOwner2 = muxInputs[1].getDefiningOp();
    if (!isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner1) ||
        !isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner2))
      return failure();
    handshake::ConditionalBranchOp suppress1 =
        cast<handshake::ConditionalBranchOp>(useOwner1);
    handshake::ConditionalBranchOp suppress2 =
        cast<handshake::ConditionalBranchOp>(useOwner2);

    Value dataOperand = muxOp.getResult();
    Value condition1 = suppress1.getConditionOperand();
    Value condition2 = suppress2.getConditionOperand();
    Value branchIn1 = suppress1.getDataOperand();
    Value branchIn2 = suppress2.getDataOperand();
    if (hasRealUses(suppress1.getTrueResult()) ||
        hasRealUses(suppress2.getTrueResult()))
      return failure();
    if (branchIn1.getDefiningOp() != branchIn2.getDefiningOp())
      return failure();
    handshake::ForkOp forkC =
        cast<handshake::ForkOp>(branchIn1.getDefiningOp());
    Value in = forkC.getOperand();
    if (!isa<handshake::ForkOp>(select.getDefiningOp()))
      return failure();
    handshake::ForkOp forkOp = cast<handshake::ForkOp>(select.getDefiningOp());
    handshake::NotOp notOp;
    bool replace = false;
    if (condition1.getDefiningOp() == forkOp &&
        isa<handshake::NotOp>(condition2.getDefiningOp())) {
      notOp = cast<handshake::NotOp>(condition2.getDefiningOp());
      Value operand = notOp.getOperand();
      if (operand.getDefiningOp() != forkOp)
        return failure();
      replace = true;
    }
    if (condition2.getDefiningOp() == forkOp &&
        isa<handshake::NotOp>(condition1.getDefiningOp())) {
      notOp = cast<handshake::NotOp>(condition1.getDefiningOp());
      Value operand = notOp.getOperand();
      if (operand.getDefiningOp() != forkOp)
        return failure();
      replace = true;
    }
    if (!replace)
      return failure();
    // rewriter.replaceOp(muxOp, branchIn2);
    eraseSinkUsers(suppress1.getTrueResult(), rewriter);
    eraseSinkUsers(suppress2.getTrueResult(), rewriter);
    rewriter.create<handshake::SinkOp>(forkC->getLoc(), select);
    rewriter.create<handshake::SinkOp>(forkC->getLoc(), condition1);
    rewriter.create<handshake::SinkOp>(forkC->getLoc(), condition2);

    rewriter.replaceOp(suppress1, {in, in});
    rewriter.replaceOp(suppress2, {in, in});
    rewriter.create<handshake::SinkOp>(suppress1->getLoc(), branchIn1);
    rewriter.replaceAllUsesWith(dataOperand, branchIn2);
    return success();
  }
};

struct HandshakeRewriteTermsPass
    : public dynamatic::impl::HandshakeRewriteTermsBase<
          HandshakeRewriteTermsPass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns.add<RemoveBranchMuxPairs, RemoveBranchMergePairs,
                 RemoveBranchCMergePairs,  RemoveConsecutiveForksPairs,
                 RemoveMuxBranchLoopPairs, RemoveMergeBranchLoopPairs, BranchToSupressForkPairs,
                 RemoveSuppressSuppressPairs, RemoveSupressForkPairs, RemoveForkSupressPairsMux>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::rewriteHandshakeTerms() {
  return std::make_unique<HandshakeRewriteTermsPass>();
}
