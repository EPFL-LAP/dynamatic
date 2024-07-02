#include "dynamatic/Transforms/HandshakeOptimization.h"
#include "dynamatic/Dialect/Handshake/HandshakeCanonicalize.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
using namespace mlir;
using namespace dynamatic;

namespace {

struct RemoveBranchMUXPairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    mlir::Operation *useOwner1 = nullptr;
    mlir::Operation *useOwner2 = nullptr;
    auto it = condBranchOp.getTrueResult().getUsers().begin();
    if (it != condBranchOp.getTrueResult().getUsers().end()) {
      useOwner1 = *it;
    }
    it = condBranchOp.getFalseResult().getUsers().begin();
    if (it != condBranchOp.getFalseResult().getUsers().end()) {
      useOwner2 = *it;
    }
    if (isa_and_nonnull<handshake::MuxOp>(useOwner1) and
        useOwner1 == useOwner2) {
      handshake::MuxOp muxOp = mlir::cast<handshake::MuxOp>(useOwner1);
      mlir::Value conditionBr = condBranchOp.getConditionOperand();
      mlir::Value selectMux = muxOp.getSelectOperand();
      if (conditionBr.getDefiningOp() == selectMux.getDefiningOp()) {
        mlir::Value dataOperand = condBranchOp.getDataOperand();
        rewriter.setInsertionPoint(muxOp);
        rewriter.create<handshake::SinkOp>(muxOp->getLoc(), selectMux);
        rewriter.setInsertionPoint(condBranchOp);
        rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), conditionBr);
        rewriter.replaceOp(muxOp, {dataOperand});
        rewriter.eraseOp(condBranchOp);
        return success();
      }
    }
    return failure();
  }
};

struct RemoveBranchMergePairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    mlir::Operation *useOwner1 = nullptr;
    mlir::Operation *useOwner2 = nullptr;
    auto it = condBranchOp.getTrueResult().getUsers().begin();
    if (it != condBranchOp.getTrueResult().getUsers().end()) {
      useOwner1 = *it;
    }
    it = condBranchOp.getFalseResult().getUsers().begin();
    if (it != condBranchOp.getFalseResult().getUsers().end()) {
      useOwner2 = *it;
    }

    if (isa_and_nonnull<handshake::MergeOp>(useOwner1) and
        useOwner1 == useOwner2) {
      handshake::MergeOp mergeOp = mlir::cast<handshake::MergeOp>(useOwner1);
      mlir::Value conditionBr = condBranchOp.getConditionOperand();
      mlir::Value dataOperand = condBranchOp.getDataOperand();
      rewriter.replaceOp(mergeOp, {dataOperand});
      rewriter.setInsertionPoint(condBranchOp);
      rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), conditionBr);
      rewriter.eraseOp(condBranchOp);
      return success();
    }
    return failure();
  }
};

struct RemoveBranchCMergePairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    mlir::Operation *useOwner1 = nullptr;
    mlir::Operation *useOwner2 = nullptr;
    auto it = condBranchOp.getTrueResult().getUsers().begin();
    if (it != condBranchOp.getTrueResult().getUsers().end()) {
      useOwner1 = *it;
    }

    it = condBranchOp.getFalseResult().getUsers().begin();
    if (it != condBranchOp.getFalseResult().getUsers().end()) {
      useOwner2 = *it;
    }
    if (isa_and_nonnull<handshake::ControlMergeOp>(useOwner1) and
        useOwner1 == useOwner2) {
      handshake::ControlMergeOp cMergeOp =
          mlir::cast<handshake::ControlMergeOp>(useOwner1);
      mlir::Value dataOperand = condBranchOp.getDataOperand();
      mlir::Value conditionBr = condBranchOp.getConditionOperand();
      rewriter.replaceOp(cMergeOp, {dataOperand, conditionBr});
      rewriter.eraseOp(condBranchOp);
      return success();
    }
    return failure();
  }
};

struct RemoveMUXBranchLoopPairs : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    mlir::Operation *useOwner = nullptr;
    auto it = muxOp->getUsers().begin();
    if (it != muxOp->getUsers().end()) {
      useOwner = *it;
    }
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner)) {
      handshake::ConditionalBranchOp condBranchOp =
          mlir::cast<handshake::ConditionalBranchOp>(useOwner);
      mlir::OperandRange dataOperands = muxOp.getDataOperands();
      mlir::Value first = dataOperands[0];
      mlir::Value second = dataOperands[1];
      mlir::Value resultMux = muxOp.getResult();
      mlir::Value select = muxOp.getSelectOperand();
      mlir::Value condition = condBranchOp.getConditionOperand();
      mlir::Value branchIn = condBranchOp.getDataOperand();
      mlir::Value trueResult = condBranchOp.getTrueResult();
      mlir::Value falseResult = condBranchOp.getFalseResult();
      mlir::ValueRange l = {second, second};
      if (resultMux == branchIn) {

        if (trueResult == first) {
          rewriter.setInsertionPoint(muxOp);
          rewriter.create<handshake::SinkOp>(muxOp->getLoc(), select);
          rewriter.replaceOp(muxOp, second);
          rewriter.replaceAllUsesWith(falseResult, second);
        } else if (trueResult == second) {
          rewriter.setInsertionPoint(muxOp);
          rewriter.create<handshake::SinkOp>(muxOp->getLoc(), select);
          rewriter.replaceOp(muxOp, first);
          rewriter.replaceAllUsesWith(falseResult, first);
        } else if (falseResult == first) {
          rewriter.setInsertionPoint(muxOp);
          rewriter.create<handshake::SinkOp>(muxOp->getLoc(), select);
          rewriter.replaceOp(muxOp, second);
          rewriter.replaceAllUsesWith(trueResult, second);
        } else if (falseResult == second) {
          rewriter.setInsertionPoint(muxOp);
          rewriter.create<handshake::SinkOp>(muxOp->getLoc(), select);
          rewriter.replaceOp(muxOp, first);
          rewriter.replaceAllUsesWith(trueResult, first);
        } else {
          return failure();
        }
        rewriter.setInsertionPoint(condBranchOp);
        rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), condition);
        rewriter.eraseOp(condBranchOp);
        return success();
      }
    }
    return failure();
  }
};

struct RemoveMergeBranchLoopPairs
    : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
    mlir::Operation *useOwner = nullptr;
    auto it = mergeOp->getUsers().begin();
    if (it != mergeOp->getUsers().end()) {
      useOwner = *it;
    }
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner)) {
      handshake::ConditionalBranchOp condBranchOp =
          mlir::cast<handshake::ConditionalBranchOp>(useOwner);

      mlir::OperandRange dataOperands = mergeOp.getDataOperands();
      mlir::Value first = (dataOperands[0]);
      mlir::Value second = (dataOperands[1]);
      mlir::Value resultMerge = mergeOp.getResult();
      mlir::Value condition = condBranchOp.getConditionOperand();
      mlir::Value branchIn = condBranchOp.getDataOperand();
      mlir::Value trueResult = condBranchOp.getTrueResult();
      mlir::Value falseResult = condBranchOp.getFalseResult();

      if (resultMerge == branchIn) {
        if (trueResult == first) {
          rewriter.replaceOp(mergeOp, second);
          rewriter.replaceAllUsesWith(falseResult, second);
        } else if (trueResult == second) {
          rewriter.replaceOp(mergeOp, first);
          rewriter.replaceAllUsesWith(falseResult, first);
        } else if (falseResult == first) {
          rewriter.replaceOp(mergeOp, second);
          rewriter.replaceAllUsesWith(trueResult, second);
        } else if (falseResult == second) {
          rewriter.replaceOp(mergeOp, first);
          rewriter.replaceAllUsesWith(trueResult, first);
        } else {
          return failure();
        }
        rewriter.setInsertionPoint(condBranchOp);
        rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), condition);
        rewriter.eraseOp(condBranchOp);
        return success();
      }
    }
    return failure();
  }
};

struct RemoveCMergeBranchLoopPairs
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    mlir::Block *parentBlock = cmergeOp->getBlock();
    auto l = parentBlock->getArguments();
    if (l.empty()) {
      return failure();
    }
    mlir::BlockArgument start = l.back();
    if (!isa<NoneType>(start.getType()))
      return failure();
    mlir::Operation *useOwner = nullptr;
    auto it = cmergeOp->getUsers().begin();
    if (it != cmergeOp->getUsers().end()) {
      useOwner = *it;
    }
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner)) {
      handshake::ConditionalBranchOp condBranchOp =
          mlir::cast<handshake::ConditionalBranchOp>(useOwner);

      mlir::OperandRange dataOperands = cmergeOp.getDataOperands();
      mlir::Value first = dataOperands[0];
      mlir::Value second = dataOperands[1];
      mlir::Value resultCMerge = cmergeOp.getResult();
      mlir::Value index = cmergeOp.getIndex();
      mlir::Value condition = condBranchOp.getConditionOperand();
      mlir::Value branchIn = condBranchOp.getDataOperand();
      mlir::Value trueResult = condBranchOp.getTrueResult();
      mlir::Value falseResult = condBranchOp.getFalseResult();

      if (resultCMerge == branchIn) {
        mlir::Value source =
            rewriter.create<handshake::SourceOp>(condBranchOp->getLoc());
        int64_t constantValue = 1;
        mlir::Type constantType = rewriter.getIntegerType(1);
        mlir::Value constantOne = rewriter.create<handshake::ConstantOp>(
            condBranchOp->getLoc(), constantType,
            rewriter.getIntegerAttr(constantType, constantValue), source);
        mlir::Value zero = rewriter.create<handshake::ConstantOp>(
            cmergeOp->getLoc(), constantType,
            rewriter.getIntegerAttr(constantType, 0), start);
        mlir::Value one = rewriter.create<handshake::ConstantOp>(
            cmergeOp->getLoc(), constantType,
            rewriter.getIntegerAttr(constantType, 1), start);
        handshake::NotOp notOp =
            rewriter.create<handshake::NotOp>(cmergeOp->getLoc(), condition);
        mlir::Value valueOfConstant;
        mlir::Value inputToMerge;
        if (trueResult == first) {
          valueOfConstant = one;
          inputToMerge = condition;
          rewriter.replaceAllUsesWith(falseResult, second);
          rewriter.replaceAllUsesWith(resultCMerge, second);
        } else if (falseResult == second) {
          valueOfConstant = zero;
          inputToMerge = condition;
          rewriter.replaceAllUsesWith(trueResult, first);
          rewriter.replaceAllUsesWith(resultCMerge, first);
        } else if (trueResult == second) {
          inputToMerge = notOp.getResult();
          valueOfConstant = zero;
          rewriter.replaceAllUsesWith(falseResult, first);
          rewriter.replaceAllUsesWith(resultCMerge, first);
        } else if (falseResult == first) {
          inputToMerge = notOp.getResult();
          valueOfConstant = one;
          rewriter.replaceAllUsesWith(trueResult, second);
          rewriter.replaceAllUsesWith(resultCMerge, second);
        } else {
          return failure();
        }
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
    }
    return failure();
  }
};

struct RemoveConsecutiveForksPairs
    : public OpRewritePattern<handshake::ForkOp> {
  using OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp fork1,
                                PatternRewriter &rewriter) const override {
    mlir::Operation *useOwner = nullptr;
    mlir::Value input = fork1.getOperand();
    mlir::ResultRange::user_range forkUsers = fork1->getUsers();
    int NumNewResultsReqd = 0;
    int NumResultsFork = fork1.getNumResults();
    handshake::ForkOp forkNext;
    for (auto it = forkUsers.begin(); it != forkUsers.end(); it++) {
      useOwner = *it;
      if (isa_and_nonnull<handshake::ForkOp>(useOwner)) {
        forkNext = mlir::cast<handshake::ForkOp>(useOwner);
        NumNewResultsReqd += forkNext->getNumResults();
        NumResultsFork--;
      }
    }
    if (NumNewResultsReqd == 0)
      return failure();
    handshake::ForkOp newFork = rewriter.create<handshake::ForkOp>(
        fork1.getLoc(), input, NumResultsFork + NumNewResultsReqd);

    mlir::ResultRange newForkResults = newFork->getResults();

    int curNew = 0;
    int curOld = 0;

    mlir::ResultRange forkResults = fork1.getResults();
    for (auto it = forkUsers.begin(); it != forkUsers.end(); it++) {
      useOwner = *it;
      if (isa_and_nonnull<handshake::ForkOp>(useOwner)) {
        forkNext = mlir::cast<handshake::ForkOp>(useOwner);
        mlir::ResultRange forkNextResults = forkNext.getResults();
        for (unsigned j{0}; j < forkNext->getNumResults(); j++) {
          rewriter.replaceAllUsesWith(forkNextResults[j],
                                      newForkResults[curNew++]);
        }
        rewriter.eraseOp(forkNext);
      } else {
        rewriter.replaceAllUsesWith(forkResults[curOld],
                                    newForkResults[curNew++]);
      }
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

    mlir::Value trueResult = condBranchOp.getTrueResult();
    mlir::Value falseResult = condBranchOp.getFalseResult();
    if (hasRealUses(trueResult))
      return failure();
    mlir::Operation *useOwner = nullptr;
    auto op = falseResult.getUsers().begin();
    if (op == falseResult.getUsers().end())
      return failure();
    // llvm::errs() << "\n\n\nSo, false value has some use\n\n";
    useOwner = *op;
    if (isa_and_nonnull<handshake::ForkOp>(useOwner)) {
      llvm::errs() << "\n\n\nGoing into fork\n\n\n\n";
      handshake::ForkOp forkOp = mlir::cast<handshake::ForkOp>(useOwner);
      int numOfResults = forkOp->getNumResults();
      llvm::errs() << numOfResults << "\n\n";
      mlir::ResultRange forkResult = forkOp->getResults();
      mlir::Value dataOperand = condBranchOp.getDataOperand();
      mlir::Value condBr = condBranchOp.getConditionOperand();
      handshake::ForkOp forkForDataop = rewriter.create<handshake::ForkOp>(
          condBranchOp.getLoc(), dataOperand, numOfResults);
      rewriter.setInsertionPointAfter(forkForDataop);
      handshake::ForkOp forkForCondition = rewriter.create<handshake::ForkOp>(
          condBranchOp.getLoc(), condBr, numOfResults);
      llvm::errs() << "New forks created\n\n\n\n";
      mlir::ResultRange forkDataResult = forkForDataop->getResults();
      mlir::ResultRange forkCondResult = forkForCondition->getResults();
      std::vector<handshake::ConditionalBranchOp> vectorForSuppress(numOfResults);
      std::vector<handshake::SinkOp> vectorForSinks(numOfResults);
      ValueRange op = {dataOperand, dataOperand};
      rewriter.replaceOp(condBranchOp, op);
      rewriter.setInsertionPointAfter(forkForCondition);
      for (int i{0}; i < numOfResults; i++) {
        llvm::errs() << i << "\n\n\n";
        vectorForSuppress[i] =
            rewriter.create<handshake::ConditionalBranchOp>(
                forkOp.getLoc(), forkCondResult[i], forkDataResult[i]);
        rewriter.setInsertionPointAfter(vectorForSuppress[i]);
        vectorForSinks[i] = rewriter.create<handshake::SinkOp>(
            vectorForSuppress[i].getLoc(),
            vectorForSuppress[i].getTrueResult());
        rewriter.replaceAllUsesWith(forkResult[i],
                                    vectorForSuppress[i].getFalseResult());
        rewriter.setInsertionPointAfter(vectorForSinks[i]);
      }
      return success();
    }
    return failure();
  }
};

struct RemoveSupressSupressPairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    mlir::Value trueResult = condBranchOp.getTrueResult();
    mlir::Value falseResult = condBranchOp.getFalseResult();
    if (hasRealUses(trueResult))
      return failure();
    mlir::Operation *useOwner = nullptr;
    auto op = falseResult.getUsers().begin();
    if (op == falseResult.getUsers().end())
      return failure();
    useOwner = *op;
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner)) {
      handshake::ConditionalBranchOp suppress2 =
          mlir::cast<handshake::ConditionalBranchOp>(useOwner);
      mlir::Value trueResult2 = suppress2.getTrueResult();
      if (hasRealUses(trueResult2))
        return failure();
      mlir::Value dataOperand = condBranchOp.getDataOperand();
      mlir::Value condBr1 = condBranchOp.getConditionOperand();
      mlir::Value condBr2 = suppress2.getConditionOperand();
      mlir::Value source =
          rewriter.create<handshake::SourceOp>(condBranchOp->getLoc());
      int64_t constantValue = 1;
      mlir::Type constantType = rewriter.getIntegerType(1);
      mlir::Value constantOne = rewriter.create<handshake::ConstantOp>(
          condBranchOp->getLoc(), constantType,
          rewriter.getIntegerAttr(constantType, constantValue), source);
      ValueRange muxOperands = {constantOne, condBr2};
      handshake::MuxOp mux = rewriter.create<handshake::MuxOp>(
          condBranchOp->getLoc(), condBr1, muxOperands);
      mlir::Value result = mux.getResult();
      ValueRange operands = {result, dataOperand};
      suppress2->setOperands(operands);
      rewriter.setInsertionPointAfter(suppress2);
      rewriter.create<handshake::SinkOp>(suppress2.getLoc(),
                                         suppress2.getTrueResult());
      ValueRange op = {dataOperand, dataOperand};
      rewriter.replaceOp(condBranchOp, op);
      return success();
    }
    return failure();
  }
};

struct BranchToSupressForkPairs
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {
    mlir::Value dataOperand = condBranchOp.getDataOperand();
    mlir::Value condition = condBranchOp.getConditionOperand();
    mlir::Value falseResult = condBranchOp.getFalseResult();
    mlir::Value trueResult = condBranchOp.getTrueResult();
    if (hasRealUses(falseResult) and hasRealUses(trueResult)) {
      handshake::ForkOp fork_data = rewriter.create<handshake::ForkOp>(
          condBranchOp->getLoc(), dataOperand, 2);
      rewriter.setInsertionPointAfter(fork_data);
      handshake::ForkOp fork_cond = rewriter.create<handshake::ForkOp>(
          condBranchOp->getLoc(), condition, 2);
      rewriter.setInsertionPointAfter(fork_cond);
      handshake::NotOp notOp = rewriter.create<handshake::NotOp>(
          condBranchOp->getLoc(), fork_cond->getResults()[1]);
      rewriter.setInsertionPointAfter(notOp);
      handshake::ConditionalBranchOp suppress1 =
          rewriter.create<handshake::ConditionalBranchOp>(
              condBranchOp->getLoc(), fork_cond->getResults()[0],
              fork_data.getResults()[0]);
      rewriter.setInsertionPointAfter(suppress1);
      handshake::SinkOp a = rewriter.create<handshake::SinkOp>(
          condBranchOp->getLoc(), suppress1.getTrueResult());
      rewriter.setInsertionPointAfter(a);
      handshake::ConditionalBranchOp suppress2 =
          rewriter.create<handshake::ConditionalBranchOp>(
              condBranchOp->getLoc(), notOp.getResult(),
              fork_data->getResults()[1]);
      rewriter.setInsertionPointAfter(suppress2);
      handshake::SinkOp b = rewriter.create<handshake::SinkOp>(
          condBranchOp->getLoc(), suppress2.getTrueResult());
      ValueRange results = {suppress1.getFalseResult(),
                            suppress2.getFalseResult()};
      rewriter.replaceOp(condBranchOp, results);
      return success();
    }
    return failure();
  }
};

struct RemoveForkSupressPairsMUX : public OpRewritePattern<handshake::ForkOp> {
  using OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    mlir::Value dataOperand = forkOp.getOperand();
    if (forkOp->getNumResults() == 2) {
      mlir::Operation *useOwner = nullptr;
      handshake::ConditionalBranchOp suppress1 = nullptr;
      handshake::ConditionalBranchOp suppress2 = nullptr;
      mlir::ResultRange::user_range users = forkOp->getUsers();

      for (auto it = users.begin(); it != users.end(); it++) {
        useOwner = *it;
        if (not(isa<handshake::ConditionalBranchOp>(useOwner)))
          return failure();
        if (suppress1) {
          suppress2 = mlir::cast<handshake::ConditionalBranchOp>(useOwner);
        } else {
          suppress1 = mlir::cast<handshake::ConditionalBranchOp>(useOwner);
        }
      }
      if (hasRealUses(suppress1.getTrueResult()) or
          hasRealUses(suppress2.getTrueResult()))
        return failure();
      mlir::Operation *op1, *op2;
      auto it = (suppress1.getFalseResult().getUsers().begin());
      if (it != suppress1.getFalseResult().getUsers().end()) {
        op1 = *it;
      }
      it = (suppress2.getFalseResult().getUsers().begin());
      if (it != suppress2.getFalseResult().getUsers().end()) {
        op2 = *it;
      }
      if (llvm::isa_and_nonnull<handshake::MuxOp>(op1) and op1 == op2) {
        handshake::MuxOp mux = mlir::cast<handshake::MuxOp>(op1);
        mlir::Value select = mux.getSelectOperand();
        mlir::Value cond1 = suppress1.getConditionOperand();
        mlir::Value cond2 = suppress2.getConditionOperand();
        mlir::Operation *f1 = cond1.getDefiningOp();
        mlir::Operation *f2 = cond2.getDefiningOp();
        mlir::Operation *m = select.getDefiningOp();
        handshake::NotOp notOp;
        handshake::ForkOp forkC;
        bool replacement = false;
        if (llvm::isa_and_nonnull<handshake::ForkOp>(m)) {
          forkC = mlir::cast<handshake::ForkOp>(m);
          mlir::Value condition = forkC.getOperand();
          mlir::Value operand;
          if (f1 == m and llvm::isa_and_nonnull<handshake::NotOp>(f2)) {
            notOp = mlir::cast<handshake::NotOp>(f2);
            operand = notOp.getOperand();
            if (operand.getDefiningOp() == forkC) {
              replacement = true;
            }
          } else if (f2 == m and llvm::isa_and_nonnull<handshake::NotOp>(f1)) {
            notOp = mlir::cast<handshake::NotOp>(f1);
            operand = notOp.getOperand();
            if (operand.getDefiningOp() == forkC) {
              replacement = true;
            }
          }
          if (replacement) {
            rewriter.replaceOp(mux, dataOperand);
            rewriter.create<handshake::SinkOp>(mux->getLoc(), condition);
            eraseSinkUsers(suppress1.getTrueResult(), rewriter);
            eraseSinkUsers(suppress2.getTrueResult(), rewriter);
            rewriter.replaceOp(suppress1, {dataOperand, dataOperand});
            rewriter.replaceOp(suppress2, {dataOperand, dataOperand});
            return success();
          }
        }
      }
    }
    return failure();
  }
};

struct HandshakeOptimizationPass
    : public dynamatic::impl::HandshakeOptimizationBase<
          HandshakeOptimizationPass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    mlir::ModuleOp mod = getOperation();

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns.add<RemoveConsecutiveForksPairs, BranchToSupressForkPairs>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeOptimization() {
  return std::make_unique<HandshakeOptimizationPass>();
}