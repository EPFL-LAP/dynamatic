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

#include "experimental/Transforms/HandshakeCombineSteeringLogic.h"
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

using namespace mlir;
using namespace dynamatic;

namespace {

/// Combine redundant init merges. These merges have one constant input and a
/// condition input. If two merges are identical, then one of them can be
/// removed
struct CombineInits : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {

    // Work only with merges having two inputs
    if (mergeOp->getNumOperands() != 2)
      return failure();

    // One of the inputs of the merge must be a constants
    int constIdx = -1;
    for (int i = 0; i < 2; i++) {
      if (isa_and_nonnull<handshake::ConstantOp>(
              mergeOp.getDataOperands()[i].getDefiningOp()))
        constIdx = i;
    }

    if (constIdx == -1)
      return failure();

    // Get the index of the other input
    int loopIdx = 1 - constIdx;

    // If there are other merges fed from the same input at the loopIdx
    DenseSet<handshake::MergeOp> redundantInits;
    for (auto *user : mergeOp.getDataOperands()[loopIdx].getUsers())
      if (isa_and_nonnull<handshake::MergeOp>(user) && user != mergeOp) {
        handshake::MergeOp mergeUser = cast<handshake::MergeOp>(user);
        if (isa_and_nonnull<handshake::ConstantOp>(
                mergeUser.getDataOperands()[constIdx].getDefiningOp()))
          redundantInits.insert(mergeUser);
      }

    if (redundantInits.empty())
      return failure();

    for (auto init : redundantInits) {
      rewriter.replaceAllUsesWith(init.getResult(), mergeOp.getResult());
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
  for (auto cons : cast<handshake::MuxOp>(op).getResult().getUsers()) {
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
    // erased using the RemoveUnusedOp<handshake::ConditionalBranchOp>
    for (auto mux : redundantMuxes) {
      rewriter.replaceAllUsesWith(mux.getResult(), muxOp.getResult());
      rewriter.eraseOp(mux);
    }

    return success();
  }
};

/// Check if two values are functionally equivalent:
///   - Same SSA value, OR
///   - Both are ConstantOps with the same attribute value, OR
///   - Both are NotOps whose inputs are themselves equivalent (recursive)
static bool areEquivalentValues(Value a, Value b) {
  if (a == b)
    return true;

  Operation *defA = a.getDefiningOp();
  Operation *defB = b.getDefiningOp();
  if (!defA || !defB)
    return false;

  if (auto constA = dyn_cast<handshake::ConstantOp>(defA)) {
    if (auto constB = dyn_cast<handshake::ConstantOp>(defB))
      return constA.getValueAttr() == constB.getValueAttr();
    return false;
  }

  if (auto notA = dyn_cast<handshake::NotOp>(defA)) {
    if (auto notB = dyn_cast<handshake::NotOp>(defB))
      return areEquivalentValues(notA.getOperand(), notB.getOperand());
    return false;
  }

  return false;
}

/// Combine MuxOps that have functionally identical inputs.
struct CombineEquivalentMuxes : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {

    if (muxOp.getNumOperands() != 3)
      return failure();

    SmallVector<handshake::MuxOp> redundant;

    muxOp->getParentRegion()->walk([&](handshake::MuxOp otherMux) {
      if (otherMux == muxOp)
        return;
      if (otherMux.getNumOperands() != 3)
        return;
      if (!areEquivalentValues(otherMux.getSelectOperand(),
                               muxOp.getSelectOperand()))
        return;
      if (!areEquivalentValues(otherMux.getDataOperands()[0],
                               muxOp.getDataOperands()[0]))
        return;
      if (!areEquivalentValues(otherMux.getDataOperands()[1],
                               muxOp.getDataOperands()[1]))
        return;
      redundant.push_back(otherMux);
    });

    if (redundant.empty())
      return failure();

    for (auto mux : redundant) {
      rewriter.replaceAllUsesWith(mux.getResult(), muxOp.getResult());
      rewriter.eraseOp(mux);
    }

    return success();
  }
};

/// Combine ConditionalBranchOps that have functionally identical inputs.
struct CombineEquivalentBranches
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<handshake::ConditionalBranchOp> redundant;

    condBranchOp->getParentRegion()->walk(
        [&](handshake::ConditionalBranchOp otherBr) {
          if (otherBr == condBranchOp)
            return;
          if (!areEquivalentValues(otherBr.getConditionOperand(),
                                   condBranchOp.getConditionOperand()))
            return;
          if (!areEquivalentValues(otherBr.getDataOperand(),
                                   condBranchOp.getDataOperand()))
            return;
          redundant.push_back(otherBr);
        });

    if (redundant.empty())
      return failure();

      for (auto br : redundant) {
      rewriter.replaceAllUsesWith(br.getTrueResult(),
                                  condBranchOp.getTrueResult());
      rewriter.replaceAllUsesWith(br.getFalseResult(),
                                  condBranchOp.getFalseResult());
      rewriter.eraseOp(br);
    }

    return success();
  }
};

/// Remove any op of type OpTy whose results are all unused.
template <typename OpTy>
struct RemoveUnusedOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // The pattern fails if the Op has any successors
    for (auto result : op->getResults()) {
      if (!result.use_empty())
        return failure();
    }

    rewriter.eraseOp(op);
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

    if (!isa_and_nonnull<handshake::NotOp>(condOperand.getDefiningOp()))
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

    if (!llvm::isa_and_nonnull<handshake::NotOp>(condOp))
      return failure();

    auto drivingNot = llvm::dyn_cast<handshake::NotOp>(condOp);

    rewriter.setInsertionPointAfter(condBranchOp);

    auto newBranch = rewriter.create<handshake::ConditionalBranchOp>(
        condOp->getLoc(), drivingNot.getOperand(),
        condBranchOp.getDataOperand());

    rewriter.replaceAllUsesWith(condBranchOp.getTrueResult(),
                                newBranch.getFalseResult());
    rewriter.replaceAllUsesWith(condBranchOp.getFalseResult(),
                                newBranch.getTrueResult());

    newBranch->setAttr("handshake.bb", condBranchOp->getAttr("handshake.bb"));
    rewriter.eraseOp(condBranchOp);

    return success();
  }
};

/// When a ConditionalBranch has cond == data (or they differ only by a NOT),
/// each output carries a known boolean. Replace the condition operand of any
/// downstream branch that uses these outputs as condition with a constant
/// 0 or 1, disconnecting the upstream branch's use.
struct SimplifyKnownConditionBranch
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value condOperand = condBranchOp.getConditionOperand();
    Value dataOperand = condBranchOp.getDataOperand();

    // Match three cases:
    //   1) cond == data                (direct)
    //   2) cond == not(data)           (inverted)
    //   3) data == not(cond)           (inverted)
    bool inverted = false;
    if (condOperand == dataOperand) {
      inverted = false;
    } else {
      Operation *condDef = condOperand.getDefiningOp();
      Operation *dataDef = dataOperand.getDefiningOp();
      if (isa_and_nonnull<handshake::NotOp>(condDef) &&
          condDef->getOperand(0) == dataOperand) {
        inverted = true;
      } else if (isa_and_nonnull<handshake::NotOp>(dataDef) &&
                 dataDef->getOperand(0) == condOperand) {
        inverted = true;
      } else {
        return failure();
      }
    }

    bool changed = false;

    // For a given output of the upstream branch, replace the condition
    // of all downstream branches that use it as condition with a constant.
    auto replaceDownstreamCond = [&](Value branchOutput, bool outputIsTrue) {
      // Runtime boolean value carried by branchOutput:
      //   direct:   true output -> 1,   false output -> 0
      //   inverted: true output -> 0,   false output -> 1
      bool knownCondTrue = inverted ? !outputIsTrue : outputIsTrue;

      // Collect downstream branches using branchOutput as condition
      SmallVector<handshake::ConditionalBranchOp> toSimplify;
      for (auto *user : branchOutput.getUsers()) {
        if (auto br = dyn_cast<handshake::ConditionalBranchOp>(user)) {
          if (br.getConditionOperand() == branchOutput)
            toSimplify.push_back(br);
        }
      }

      for (auto br : toSimplify) {
        rewriter.setInsertionPoint(br);

        // Create source as trigger
        auto sourceOp =
            rewriter.create<handshake::SourceOp>(br.getLoc());
        if (auto bbAttr = br->getAttr("handshake.bb"))
          sourceOp->setAttr("handshake.bb", bbAttr);

        // Build the i1 attribute
        auto i1Type = rewriter.getIntegerType(1);
        auto cstAttr =
            rewriter.getIntegerAttr(i1Type, knownCondTrue ? 1 : 0);

        // Check if the condition operand is channelified
        Type condType = branchOutput.getType();
        handshake::ConstantOp constOp;

        if (auto channelType =
                dyn_cast<handshake::ChannelType>(condType)) {
          // Channelified: use 4-arg constructor (loc, resultType, attr, ctrl)
          // matching the pattern from the existing codebase
          constOp = rewriter.create<handshake::ConstantOp>(
              br.getLoc(), channelType, cstAttr, sourceOp.getResult());
        } else {
          // Raw i1: use 3-arg constructor (loc, attr, ctrl)
          constOp = rewriter.create<handshake::ConstantOp>(
              br.getLoc(), cstAttr, sourceOp.getResult());
        }

        if (auto bbAttr = br->getAttr("handshake.bb"))
          constOp->setAttr("handshake.bb", bbAttr);

        // Replace condition operand of downstream branch
        br->setOperand(0, constOp.getResult());

        changed = true;
      }
    };

    replaceDownstreamCond(condBranchOp.getTrueResult(), /*outputIsTrue=*/true);
    replaceDownstreamCond(condBranchOp.getFalseResult(), /*outputIsTrue=*/false);

    return changed ? success() : failure();
  }
};

/// Eliminate a ConditionalBranch whose condition is a constant.
/// Short-circuit: the always-taken output is replaced with the data operand,
/// and the branch (along with its feeding constant + source) is erased.
struct EliminateConstantCondBranch
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
                                PatternRewriter &rewriter) const override {

    Value condOperand = condBranchOp.getConditionOperand();
    auto constOp =
        dyn_cast_or_null<handshake::ConstantOp>(condOperand.getDefiningOp());
    if (!constOp)
      return failure();

    auto constAttr = dyn_cast<IntegerAttr>(constOp.getValueAttr());
    if (!constAttr)
      return failure();

    bool condIsTrue = constAttr.getValue().getBoolValue();

    Value takenResult = condIsTrue ? condBranchOp.getTrueResult()
                                   : condBranchOp.getFalseResult();
    Value notTakenResult = condIsTrue ? condBranchOp.getFalseResult()
                                      : condBranchOp.getTrueResult();

    // Only proceed when the never-taken side has no users
    if (!notTakenResult.use_empty())
      return failure();

    // Short-circuit the always-taken side
    rewriter.replaceAllUsesWith(takenResult, condBranchOp.getDataOperand());

    // Erase the branch
    rewriter.eraseOp(condBranchOp);
    
    // Clean up the constant + source if they have no other users
    if (constOp.getResult().use_empty()) {
      Value trigger = constOp.getCtrl();
      rewriter.eraseOp(constOp);
      if (auto sourceOp = dyn_cast_or_null<handshake::SourceOp>(
              trigger.getDefiningOp())) {
        if (sourceOp.getResult().use_empty())
          rewriter.eraseOp(sourceOp);
      }
    }
    return success();
  }
};

/// Simple driver for the Handshake Combine Branches Merges pass, based on a
/// greedy pattern rewriter.
struct HandshakeCombineSteeringLogicPass
    : public dynamatic::experimental::ftd::impl::
          HandshakeCombineSteeringLogicBase<HandshakeCombineSteeringLogicPass> {
  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveUnusedOp<handshake::MuxOp>,
                 RemoveUnusedOp<handshake::ConditionalBranchOp>,
                 RemoveUnusedOp<handshake::ConstantOp>,
                 RemoveUnusedOp<handshake::SourceOp>,
                 RemoveUnusedOp<handshake::NotOp>,
                 CombineBranchesOppositeSign,
                 CombineInits, CombineMuxes, RemoveNotCondition,
                 SimplifyKnownConditionBranch, EliminateConstantCondBranch,
                 CombineEquivalentMuxes, CombineEquivalentBranches>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::ftd::combineSteeringLogic() {
  return std::make_unique<HandshakeCombineSteeringLogicPass>();
}
