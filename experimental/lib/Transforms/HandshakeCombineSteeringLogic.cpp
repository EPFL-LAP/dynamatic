//===- HandshakeCombineSteeringLogic.cpp - Simplify FTD  ----*- C++ -*-----===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass which simplify the resulting FTD circuit by
// merging units which have the smae inputs and the same outputs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <cassert>

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKECOMBINESTEERINGLOGIC
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;

namespace {

/// Combine redundant init merges. These merges have one constant input and a
/// condition input. If two merges are identical, then one of them can be
/// removed
struct CombineInits : public OpRewritePattern<handshake::InitOp> {
  using OpRewritePattern<handshake::InitOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::InitOp initOp,
                                PatternRewriter &rewriter) const override {

    // Work only with init having two inputs
    if (initOp->getNumOperands() != 1)
      return failure();

    // If there are other merges fed from the same input at the loopIdx
    DenseSet<handshake::InitOp> redundantInits;
    for (auto *user : initOp.getOperand().getUsers())
      if (isa_and_nonnull<handshake::InitOp>(user) && user != initOp) {
        handshake::InitOp initUser = cast<handshake::InitOp>(user);
          redundantInits.insert(initUser);
      }

    if (redundantInits.empty())
      return failure();

    for (auto init : redundantInits) {
      rewriter.replaceAllUsesWith(init.getResult(), initOp.getResult());
      rewriter.eraseOp(init);
    }

    return success();
  }
};

/// Returns true if the loop under analysis has a self regenerating mux. One
/// input of the mux comes from the mux itself, while the other input comes from
/// somewhere else.
bool isSelfRegenerateMux(handshake::MuxOp muxOp, int &muxCycleInputIdx) {

  // One user must be a Branch; otherwise, the pattern match fails
  DenseSet<handshake::ConditionalBranchOp> branches;

  for (auto *muxUser : muxOp.getResult().getUsers()) {
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(muxUser)) {
      auto br = cast<handshake::ConditionalBranchOp>(muxUser);
      branches.insert(br);
    }
  }

  // One of the conditional branches that were found should feed muxOp forming a
  // cycle
  bool foundCycle = false;
  int operIdx = 0;
  handshake::ConditionalBranchOp condBranchOp;

  for (auto muxOperand : muxOp.getDataOperands()) {
    auto *op = muxOperand.getDefiningOp();
    if (isa_and_nonnull<handshake::ConditionalBranchOp>(op)) {
      auto br = cast<handshake::ConditionalBranchOp>(op);
      if (branches.contains(br)) {
        foundCycle = true;
        muxCycleInputIdx = operIdx;
        condBranchOp = br;
        break;
      }
    }
    operIdx++;
  }

  return foundCycle;
}

// Apply DFS on the producers of a particular Mux feeding a particular input,
// returning the first non-Mux producer's result value
Value returnNonMuxProducerVal(handshake::MuxOp muxOp, int idx) {
  Value val = muxOp.getDataOperands()[idx];
  Operation *prod = val.getDefiningOp();
  if (isa_and_nonnull<handshake::MuxOp>(prod))
    return returnNonMuxProducerVal(cast<handshake::MuxOp>(prod), idx);
  return val;
}

// Apply DFS on the consumers of op until you hit a Mux in the same BB as that
// of referenceMuxOp
Operation *returnMuxAtSameDepth(Operation *op,
                                handshake::MuxOp referenceMuxOp) {
  if (!isa_and_nonnull<handshake::MuxOp>(op) || op == referenceMuxOp)
    return nullptr;

  if (op->getAttr("handshake.bb") == referenceMuxOp->getAttr("handshake.bb"))
    return op;

  // Otherwise, explore all users in DFS-like traversal until you hit a match
  Operation *finalOp = nullptr;
  for (auto *cons : cast<handshake::MuxOp>(op).getResult().getUsers()) {
    Operation *potentialOp = returnMuxAtSameDepth(cons, referenceMuxOp);
    if (potentialOp != nullptr) {
      finalOp = potentialOp;
      break;
    }
  }
  return finalOp;
}

// Note: This pattern assumes that all Muxes belonging to 1 loop have the same
// conventions about the index of the input coming from outside the loop and
// that coming from inside through a cycle
// This pattern combines all Muxes that are used to regenerate the same value
// but to different consumers.. It searches for a Mux that has a bwd edge
// (cyclic input) and searches for all Muxes using the some condition and also
// having a bwd edge
struct CombineMuxes : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {

    // The mux needs to have three inputs
    if (muxOp.getNumOperands() != 3)
      return failure();

    int muxCycleIdx;
    // Exit if it's not a self regenerate mux
    if (!isSelfRegenerateMux(muxOp, muxCycleIdx))
      return failure();

    int muxOutIdx = 1 - muxCycleIdx;

    DenseSet<handshake::MuxOp> dataMuxUsers;
    DenseSet<handshake::MuxOp> redundantMuxes;

    // Identify the first non-Mux producer of the muxOp by running a DFS-like
    // traversal and return its produced value
    Value valProducedByNonMux = returnNonMuxProducerVal(muxOp, muxOutIdx);

    // Get users of the non-Mux operation at the muxOuterInputIdx
    for (auto *dataUser : valProducedByNonMux.getUsers()) {
      Operation *returnedMux = returnMuxAtSameDepth(dataUser, muxOp);
      if (returnedMux != nullptr) {
        auto muxUser = cast<handshake::MuxOp>(returnedMux);
        int tempValue;
        if (isSelfRegenerateMux(muxOp, tempValue))
          dataMuxUsers.insert(muxUser);
      }
    }

    // Get users of the operation at the select input, and consider only the
    // users which were also in `dataMuxUsers`
    for (auto *selUser : muxOp.getSelectOperand().getUsers())
      if (isa_and_nonnull<handshake::MuxOp>(selUser) && selUser != muxOp) {
        auto muxUser = cast<handshake::MuxOp>(selUser);
        int tempValue;
        if (isSelfRegenerateMux(muxUser, tempValue) &&
            dataMuxUsers.contains(muxUser)) {
          redundantMuxes.insert(muxUser);
        }
      }

    if (redundantMuxes.empty())
      return failure();

    // Loop over redundantMuxes and replace the users of them with the output of
    // muxOp Note that the users of all redundantMuxes include the Branches
    // forming cycles with each of them, but as we erase the redundantMuxes,
    // these Branches will have their two outputs feeding nothing and will be
    // erased using the RemoveDoubleSinkBranches
    for (auto mux : redundantMuxes) {
      rewriter.replaceAllUsesWith(mux.getResult(), muxOp.getResult());
      rewriter.eraseOp(mux);
    }

    return success();
  }
};

/// Remove muxes that have no successors
struct RemoveSinkMuxes : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {

    // The pattern fails if the Mux has any successors
    if (!muxOp.getResult().getUsers().empty())
      return failure();

    rewriter.eraseOp(muxOp);
    return success();
  }
};

struct RemoveSinkInits : public OpRewritePattern<handshake::InitOp> {
  using OpRewritePattern<handshake::InitOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::InitOp initOp,
                                PatternRewriter &rewriter) const override {

    // The pattern fails if the Mux has any successors
    if (!initOp.getResult().getUsers().empty())
      return failure();

    rewriter.eraseOp(initOp);
    return success();
  }
};


/// Remove conditional branches that have no successors
struct RemoveDoubleSinkBranches
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value branchTrueResult = condBranchOp.getTrueResult();
    Value branchFalseResult = condBranchOp.getFalseResult();

    // The pattern fails if the branch has either true or false successors
    if (!branchTrueResult.getUsers().empty())
      return failure();

    if (!branchFalseResult.getUsers().empty())
      return failure();

    rewriter.eraseOp(condBranchOp);
    return success();
  }
};

static DenseSet<handshake::ConditionalBranchOp>
findRedundantBranches(Value condOperand, Value dataOperand,
                      handshake::ConditionalBranchOp originalBranch) {
  DenseSet<handshake::ConditionalBranchOp> condUsers;
  DenseSet<handshake::ConditionalBranchOp> redundantBranches;

  // Get all the users of the condition operand, and keep the branches only
  for (auto *condUser : condOperand.getUsers()) {
    if (condUser == originalBranch)
      continue;
    if (auto br = dyn_cast<handshake::ConditionalBranchOp>(condUser); br) {
      if (br.getConditionOperand() == condOperand)
        condUsers.insert(br);
    }
  }

  // Check if one of the branch users of the data input was also a user of
  // the condition input: in this case, the branch is redundant
  for (auto *dataUser : dataOperand.getUsers()) {
    if (dataUser == originalBranch)
      continue;
    if (auto br = dyn_cast<handshake::ConditionalBranchOp>(dataUser); br) {
      if (br.getDataOperand() == dataOperand && condUsers.contains(br))
        redundantBranches.insert(br);
    }
  }

  return redundantBranches;
}

/// Remove branches which have the same data operands but opposite condition
/// operand
struct CombineBranchesOppositeSign
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = condBranchOp.getConditionOperand();

    if (!isa_and_nonnull<handshake::NotIOp>(condOperand.getDefiningOp()))
      return failure();

    condOperand = condOperand.getDefiningOp()->getOperand(0);

    auto redundantBranches =
        findRedundantBranches(condOperand, dataOperand, condBranchOp);

    // Nothing to erase
    if (redundantBranches.empty())
      return failure();

    // Erase the redundant branch
    for (auto br : redundantBranches) {
      rewriter.replaceAllUsesWith(br.getFalseResult(),
                                  condBranchOp.getTrueResult());
      rewriter.replaceAllUsesWith(br.getTrueResult(),
                                  condBranchOp.getFalseResult());
      rewriter.eraseOp(br);
    }

    return success();
  }
};

/// Remove branches with same data operands and same conditional operand
struct RemoveNotCondition
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value condValue = condBranchOp.getConditionOperand();
    Operation *condOp = condValue.getDefiningOp();

    if (!llvm::isa_and_nonnull<handshake::NotIOp>(condOp))
      return failure();

    auto drivingNot = llvm::dyn_cast<handshake::NotIOp>(condOp);

    rewriter.setInsertionPointAfter(condBranchOp);

    auto newBranch = rewriter.create<handshake::ConditionalBranchOp>(
        condOp->getLoc(), drivingNot.getOperand(),
        condBranchOp.getDataOperand());

    rewriter.replaceAllUsesWith(condBranchOp.getTrueResult(),
                                newBranch.getFalseResult());
    rewriter.replaceAllUsesWith(condBranchOp.getFalseResult(),
                                newBranch.getTrueResult());

    newBranch->setAttr("handshake.bb", condBranchOp->getAttr("handshake.bb"));

    return success();
  }
};

/// Remove branches with same data operands and same conditional operand
struct CombineBranchesSameSign
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value dataOperand = condBranchOp.getDataOperand();
    Value condOperand = condBranchOp.getConditionOperand();

    auto redundantBranches =
        findRedundantBranches(condOperand, dataOperand, condBranchOp);

    // Nothing to erase
    if (redundantBranches.empty())
      return failure();

    // Erase the redundant branch
    for (auto br : redundantBranches) {
      rewriter.replaceAllUsesWith(br.getTrueResult(),
                                  condBranchOp.getTrueResult());
      rewriter.replaceAllUsesWith(br.getFalseResult(),
                                  condBranchOp.getFalseResult());
      rewriter.eraseOp(br);
    }
    return success();
  }
};

struct EraseSingleOutputDemuxes : public OpRewritePattern<handshake::DemuxOp> {
  using OpRewritePattern<handshake::DemuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::DemuxOp demuxOp,
                                PatternRewriter &rewriter) const override {
    ValueRange dataOutputs = demuxOp.getResults();
    if (dataOutputs.size() != 1)
      return failure();

    // Insert a sink to consume the demux's select token
    rewriter.setInsertionPoint(demuxOp);
    Value select = demuxOp.getSelectOperand();
    rewriter.create<handshake::SinkOp>(demuxOp->getLoc(), select);
    Value dataInput = demuxOp.getDataOperand();
    rewriter.replaceOp(demuxOp, dataInput);
    return success();
  }
};

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

struct EraseNoUsesUntagger : public OpRewritePattern<handshake::UntaggerOp> {
  using OpRewritePattern<handshake::UntaggerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::UntaggerOp untaggerOp,
                                PatternRewriter &rewriter) const override {
    ValueRange untaggerOutputs = untaggerOp.getResults();
    if (untaggerOutputs.size() != 2)
      return failure();

    bool notEmpty = false;
    for(int i = 0; i < untaggerOutputs.size(); i++) {
       for (auto *user : untaggerOutputs[i].getUsers()) {
        if(!isa_and_nonnull<handshake::SinkOp>(user)) {
          notEmpty = true;
          break;
        }
       }
    }

    if(notEmpty)
      return failure();

    // Insert a sink to consume the mux's select token
    // rewriter.setInsertionPoint(untaggerOp);
    // rewriter.create<handshake::SinkOp>(untaggerOp->getLoc(), untaggerOp.getOperand());

    rewriter.eraseOp(untaggerOp);
    return success();
  }
};

/// Simple driver for the Handshake Combine Branches Merges pass, based on a
/// greedy pattern rewriter.
struct HandshakeCombineSteeringLogicPass
    : public dynamatic::experimental::impl::HandshakeCombineSteeringLogicBase<
          HandshakeCombineSteeringLogicPass> {
  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveSinkMuxes, RemoveSinkInits, RemoveDoubleSinkBranches,
                 CombineBranchesSameSign, CombineBranchesOppositeSign,
                 CombineInits, CombineMuxes, RemoveNotCondition, EraseSingleOutputDemuxes, EraseSingleInputMuxes>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
} // namespace
