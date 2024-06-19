#include "dynamatic/Transforms/HandshakeOptimization.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
          mlir::cast<handshake::ConditionalBranchOp>(condBranchOp);

      mlir::OperandRange dataOperands = muxOp.getDataOperands();
      mlir::Value first = *(dataOperands.begin());
      mlir::Value second = *(dataOperands.begin()++);
      mlir::Value resultMux = muxOp.getResult();
      mlir::Value select = muxOp.getSelectOperand();
      mlir::Value condition = condBranchOp.getConditionOperand();
      mlir::Value branchIn = condBranchOp.getDataOperand();
      mlir::Value trueResult = condBranchOp.getTrueResult();
      mlir::Value falseResult = condBranchOp.getFalseResult();

      if (resultMux == branchIn) {
        if (trueResult == first) {
          rewriter.replaceAllUsesWith(falseResult, second);
        } else if (trueResult == second) {
          rewriter.replaceAllUsesWith(falseResult, first);
        } else if (falseResult == first) {
          rewriter.replaceAllUsesWith(trueResult, second);
        } else if (falseResult == second) {
          rewriter.replaceAllUsesWith(trueResult, first);
        } else {
          return failure();
        }
        rewriter.setInsertionPoint(condBranchOp);
        rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), condition);
        rewriter.setInsertionPoint(muxOp);
        rewriter.create<handshake::SinkOp>(muxOp->getLoc(), select);
        rewriter.eraseOp(muxOp);
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
          mlir::cast<handshake::ConditionalBranchOp>(condBranchOp);

      mlir::OperandRange dataOperands = mergeOp.getDataOperands();
      mlir::Value first = *(dataOperands.begin());
      mlir::Value second = *(dataOperands.begin()++);
      mlir::Value resultMerge = mergeOp.getResult();
      mlir::Value condition = condBranchOp.getConditionOperand();
      mlir::Value branchIn = condBranchOp.getDataOperand();
      mlir::Value trueResult = condBranchOp.getTrueResult();
      mlir::Value falseResult = condBranchOp.getFalseResult();

      if (resultMerge == branchIn) {
        if (trueResult == first) {
          rewriter.replaceAllUsesWith(falseResult, second);
        } else if (trueResult == second) {
          rewriter.replaceAllUsesWith(falseResult, first);
        } else if (falseResult == first) {
          rewriter.replaceAllUsesWith(trueResult, second);
        } else if (falseResult == second) {
          rewriter.replaceAllUsesWith(trueResult, first);
        } else {
          return failure();
        }
        rewriter.setInsertionPoint(condBranchOp);
        rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), condition);
        rewriter.eraseOp(mergeOp);
        rewriter.eraseOp(condBranchOp);
        return success();
      }
    }
    return failure();
  }
};

// struct RemoveCMergeBranchLoopPairs
//     : public OpRewritePattern<handshake::ControlMergeOp> {
//   using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
//                                 PatternRewriter &rewriter) const override {
//     mlir::Operation *useOwner = nullptr;
//     auto it = cmergeOp->getUsers().begin();
//     if (it != cmergeOp->getUsers().end()) {
//       useOwner = *it;
//     }
//     if (isa_and_nonnull<handshake::ConditionalBranchOp>(useOwner)) {
//       handshake::ConditionalBranchOp condBranchOp =
//           mlir::cast<handshake::ConditionalBranchOp>(condBranchOp);

//       mlir::OperandRange dataOperands = cmergeOp.getDataOperands();
//       mlir::Value first = *(dataOperands.begin());
//       mlir::Value second = *(dataOperands.begin()++);
//       mlir::Value resultCMerge = cmergeOp.getResult();
//       mlir::Value condition = condBranchOp.getConditionOperand();
//       mlir::Value branchIn = condBranchOp.getDataOperand();
//       mlir::Value trueResult = condBranchOp.getTrueResult();
//       mlir::Value falseResult = condBranchOp.getFalseResult();

//       if (resultCMerge == branchIn) {
//         if (trueResult == first) {
//           rewriter.replaceAllUsesWith(falseResult, second);
//         } else if (trueResult == second) {
//           rewriter.replaceAllUsesWith(falseResult, first);
//         } else if (falseResult == first) {
//           rewriter.replaceAllUsesWith(trueResult, second);
//         } else if (falseResult == second) {
//           rewriter.replaceAllUsesWith(trueResult, first);

//         } else {
//           return failure();
//         }
//         rewriter.setInsertionPoint(condBranchOp);
//         rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), condition);
//         rewriter.eraseOp(mergeOp);
//         rewriter.eraseOp(condBranchOp);
//         return success();
//       }
//     }
//     return failure();
//   }
// };

// struct RemoveConsecutiveForksPairs
//     : public OpRewritePattern<handshake::ForkOp> {
//   using OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(handshake::ForkOp Fork1,
//                                 PatternRewriter &rewriter) const override {
//     mlir::Operation *useOwner = nullptr;
//     Fork1->getUsers();
//     auto it = Fork1->getUsers().begin();
//     for (auto it = Fork1->getUsers().begin(); it != Fork1->getUsers(.end))

//       mlir::OperandRange dataOperands = cmergeOp.getDataOperands();
//     mlir::Value first = *(dataOperands.begin());
//     mlir::Value second = *(dataOperands.begin()++);
//     mlir::Value resultCMerge = cmergeOp.getResult();
//     mlir::Value condition = condBranchOp.getConditionOperand();
//     mlir::Value branchIn = condBranchOp.getDataOperand();
//     mlir::Value trueResult = condBranchOp.getTrueResult();
//     mlir::Value falseResult = condBranchOp.getFalseResult();

//     if (resultCMerge == branchIn) {
//       if (trueResult == first) {
//         rewriter.replaceAllUsesWith(falseResult, second);

//       } else if (trueResult == second) {
//         rewriter.replaceAllUsesWith(falseResult, first);
//       } else if (falseResult == first) {
//         rewriter.replaceAllUsesWith(trueResult, second);
//       } else if (falseResult == second) {
//         rewriter.replaceAllUsesWith(trueResult, first);

//       } else {
//         return failure();
//       }
//       rewriter.setInsertionPoint(condBranchOp);
//       rewriter.create<handshake::SinkOp>(condBranchOp->getLoc(), condition);
//       rewriter.eraseOp(mergeOp);
//       rewriter.eraseOp(condBranchOp);
//       return success();
//     }
//   }
//   return failure();
// }
// };
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
    patterns.add<RemoveBranchCMergePairs, RemoveBranchMUXPairs,
                 RemoveBranchMergePairs, RemoveMergeBranchLoopPairs,
                 RemoveMUXBranchLoopPairs>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}
; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeOptimization() {
  return std::make_unique<HandshakeOptimizationPass>();
}