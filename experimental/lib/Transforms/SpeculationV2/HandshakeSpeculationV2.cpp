//===- HandshakeSpeculationV2.cpp - Speculative Dataflows -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Placement of Speculation components to enable speculative execution.
//
//===----------------------------------------------------------------------===//

#include "HandshakeSpeculationV2.h"
#include "MaterializationUtil.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <fstream>

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculationv2;

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

// std::unique_ptr<dynamatic::DynamaticPass> createHandshakeSpeculationV2();

#define GEN_PASS_DEF_HANDSHAKESPECULATIONV2
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct HandshakeSpeculationV2Pass
    : public dynamatic::experimental::speculationv2::impl::
          HandshakeSpeculationV2Base<HandshakeSpeculationV2Pass> {
  using HandshakeSpeculationV2Base<
      HandshakeSpeculationV2Pass>::HandshakeSpeculationV2Base;

  void runDynamaticPass() override;
};

/// Returns whether the loop condition is inverted (i.e., false continues the
/// loop).
static FailureOr<bool> isLoopConditionInverted(FuncOp &funcOp,
                                               unsigned loopHeadBB,
                                               unsigned loopTailBB) {
  // Find ConditionBranchOp in the loop tail BB
  ConditionalBranchOp condBrOp = nullptr;
  for (auto condBrCandidate : funcOp.getOps<ConditionalBranchOp>()) {
    auto condBB = getLogicBB(condBrCandidate);
    if (condBB && *condBB == loopTailBB) {
      condBrOp = condBrCandidate;
      break;
    }
  }

  if (!condBrOp)
    return funcOp.emitError(
        "Could not find ConditionalBranchOp in loop tail BB");

  std::optional<unsigned> trueResultBB =
      getLogicBB(getUniqueUser(condBrOp.getTrueResult()));
  std::optional<unsigned> falseResultBB =
      getLogicBB(getUniqueUser(condBrOp.getTrueResult()));
  if (trueResultBB && *trueResultBB == loopHeadBB) {
    // The condition is not inverted.
    return false;
  }
  if (falseResultBB && *falseResultBB == loopHeadBB) {
    // The condition is inverted.
    return true;
  }
  return funcOp.emitError("Either true or false result of ConditionalBranchOp "
                          "is not the loop backedge.");
}

/// Replaces all branches in the specified BB with passers, and returns the
/// passer control values for trueValue and falseValue.
/// Potentially can be applied to branches inside loops (i.e. PMSC), or to those
/// not at the loop's bottom in cases with multiple loop exits.
static FailureOr<std::pair<Value, Value>>
replaceBranchesWithPassers(FuncOp &funcOp, unsigned bb) {
  // Find one ConditionBranchOp in the BB
  ConditionalBranchOp referenceBranch = nullptr;
  for (auto candidate : funcOp.getOps<ConditionalBranchOp>()) {
    auto condBB = getLogicBB(candidate);
    if (condBB && *condBB == bb) {
      referenceBranch = candidate;
      break;
    }
  }

  if (!referenceBranch)
    return funcOp.emitError("Could not find ConditionalBranchOp in the BB: " +
                            std::to_string(bb));

  Value condition = referenceBranch.getConditionOperand();

  // Build a NotOp to invert the condition
  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(referenceBranch);
  NotOp invertCondition = builder.create<NotOp>(condition.getLoc(), condition);
  setBB(invertCondition, bb);

  // Replace all branches in the BB with passers
  for (auto branch :
       llvm::make_early_inc_range(funcOp.getOps<ConditionalBranchOp>())) {
    if (getLogicBB(branch) != bb)
      continue;

    // The condition must be the same (ignoring the difference of the fork
    // outputs)
    if (!equalsIndirectly(condition, branch.getConditionOperand()))
      return branch.emitError("Branch condition does not match the condition "
                              "of the reference branch");

    Value data = branch.getDataOperand();

    builder.setInsertionPoint(branch);

    // Build a passer for the trueResult
    PasserOp trueResultPasser =
        builder.create<PasserOp>(branch.getLoc(), data, condition);
    setBB(trueResultPasser, bb);
    branch.getTrueResult().replaceAllUsesWith(trueResultPasser.getResult());

    // Build a passer for the falseResult
    // The passer ctrl is inverted condition.
    PasserOp falseResultPasser = builder.create<PasserOp>(
        branch.getLoc(), data, invertCondition.getResult());
    setBB(falseResultPasser, bb);
    branch.getFalseResult().replaceAllUsesWith(falseResultPasser.getResult());

    // Erase the branch
    branch->erase();
  }

  return std::pair<Value, Value>{condition, invertCondition.getResult()};
}

/// Replace the CMerge-controlled loop header with Init[False]-controlled one.
static FailureOr<Value> updateLoopHeader(FuncOp &funcOp, unsigned loopHeadBB,
                                         unsigned loopTailBB,
                                         Value loopContinue) {
  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(funcOp.getBodyBlock(),
                            funcOp.getBodyBlock()->begin());

  // Build an InitOp[False]
  InitOp initOp =
      builder.create<InitOp>(loopContinue.getLoc(), loopContinue, 0);
  setBB(initOp, loopHeadBB);

  // Find control merge in the loop head BB.
  ControlMergeOp cmergeOp = nullptr;
  for (auto cmergeCandidate :
       llvm::make_early_inc_range(funcOp.getOps<handshake::ControlMergeOp>())) {
    if (getLogicBB(cmergeCandidate) != loopHeadBB)
      continue;

    if (cmergeOp)
      return funcOp.emitError(
          "Multiple ControlMergeOps found in the loop head BB");

    cmergeOp = cmergeCandidate;
  }

  // Only support basic blocks with two predecessors.
  assert(cmergeOp.getDataOperands().size() == 2 &&
         "The loop head BB must have exactly two predecessors");

  // The backedge must be the second operand. If it is the first operand, we
  // need to swap operands.
  bool needsSwapping;
  Operation *definingOp0 = cmergeOp.getDataOperands()[0].getDefiningOp();
  Operation *definingOp1 = cmergeOp.getDataOperands()[1].getDefiningOp();
  Value entry, backedge;
  if (definingOp0 && getLogicBB(definingOp0) == loopTailBB) {
    needsSwapping = true;
    entry = cmergeOp.getDataOperands()[1];
    backedge = cmergeOp.getDataOperands()[0];
  } else if (definingOp1 && getLogicBB(definingOp1) == loopTailBB) {
    needsSwapping = false;
    entry = cmergeOp.getDataOperands()[0];
    backedge = cmergeOp.getDataOperands()[1];
  } else {
    return cmergeOp.emitError(
        "Expected one of the operands to be defined in the loop tail BB");
  }

  // Update muxes
  for (auto muxOp : funcOp.getOps<handshake::MuxOp>()) {
    if (getLogicBB(muxOp) != loopHeadBB)
      continue;

    assert(muxOp.getDataOperands().size() == 2);

    // Update the select operand
    muxOp.getSelectOperandMutable()[0].set(initOp.getResult());

    if (needsSwapping) {
      // Swap operands
      Value entry = muxOp.getDataOperands()[1];
      muxOp.getDataOperandsMutable()[1].set(muxOp.getDataOperands()[0]);
      muxOp.getDataOperandsMutable()[0].set(entry);
    }
  }

  // Build a MuxOp to replace the CMergeOp
  // Use the result of the init as the selector.
  builder.setInsertionPoint(cmergeOp);
  MuxOp muxOp = builder.create<MuxOp>(
      cmergeOp.getLoc(), cmergeOp.getResult().getType(),
      /*selector=*/initOp.getResult(), llvm::ArrayRef{entry, backedge});
  setBB(muxOp, loopHeadBB);
  cmergeOp.getResult().replaceAllUsesWith(muxOp.getResult());

  // Erase CMerge (and possibly connected fork)
  eraseMaterializedOperation(cmergeOp);

  return initOp.getResult();
}

/// Appends a repeating init to the value. Returns the result of the repeating
/// init.
static Value appendRepeatingInit(Value val) {
  OpBuilder builder(val.getContext());
  builder.setInsertionPoint(val.getDefiningOp());

  SpecV2RepeatingInitOp repeatingInitOp =
      builder.create<SpecV2RepeatingInitOp>(val.getLoc(), val, 1);
  inheritBB(val.getDefiningOp(), repeatingInitOp);

  return repeatingInitOp.getResult();
}

/// Appends an init to the value. Returns the result of the init.
static Value appendInit(Value val) {
  OpBuilder builder(val.getContext());
  builder.setInsertionPoint(val.getDefiningOp());

  InitOp initOp = builder.create<InitOp>(val.getLoc(), val, 0);
  inheritBB(val.getDefiningOp(), initOp);

  return initOp.getResult();
}

/// Returns if the circuit is eligible for MuxPasserSwap.
static bool isEligibleForMuxPasserSwap(MuxOp muxOp, Value newSelector,
                                       Value newSpecLoopContinue) {
  // Ensure the rewritten subcircuit structure.
  Operation *backedgeDefiningOp = muxOp.getDataOperands()[1].getDefiningOp();
  if (!isa<PasserOp>(backedgeDefiningOp))
    return false;
  auto passerOp = cast<PasserOp>(backedgeDefiningOp);

  // Ensure the context
  Operation *selectorDefiningOp =
      getIndirectDefiningOp(muxOp.getSelectOperand());
  if (!isa<InitOp>(selectorDefiningOp))
    return false;
  auto initOp = cast<InitOp>(selectorDefiningOp);

  if (!equalsIndirectly(passerOp.getCtrl(), initOp.getOperand()))
    return false;

  Operation *newSelectorDefiningOp = getIndirectDefiningOp(newSelector);
  if (!isa<InitOp>(newSelectorDefiningOp))
    return false;
  auto newInitOp = cast<InitOp>(newSelectorDefiningOp);

  if (!equalsIndirectly(newInitOp.getOperand(), newSpecLoopContinue))
    return false;

  Operation *newSpecLoopContinueDefiningOp =
      getIndirectDefiningOp(newSpecLoopContinue);
  if (!isa<SpecV2RepeatingInitOp>(newSpecLoopContinueDefiningOp))
    return false;
  auto newRepeatingInitOp =
      cast<SpecV2RepeatingInitOp>(newSpecLoopContinueDefiningOp);
  if (!equalsIndirectly(newRepeatingInitOp.getOperand(), initOp.getOperand()))
    return false;

  // TODO: Verify token counts

  return true;
}

/// Performs the MuxPasserSwap, swapping the MuxOp and PasserOp, and updating
/// the select operand and control of the MuxOp and PasserOp.
/// Returns the PasserOp that was swapped.
static PasserOp performMuxPasserSwap(MuxOp muxOp, Value newSelector,
                                     Value newSpecLoopContinue) {
  auto passerOp =
      dyn_cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp());

  // Materialization is required for swapping
  assertMaterialization(passerOp.getResult());

  // Swap mux and passer
  muxOp.getDataOperandsMutable()[1].set(passerOp.getData());
  passerOp.getDataMutable()[0].set(muxOp.getResult());
  muxOp.getResult().replaceAllUsesExcept(passerOp.getResult(), passerOp);

  // Update the select operand and control
  muxOp.getSelectOperandMutable()[0].set(newSelector);
  passerOp.getCtrlMutable()[0].set(newSpecLoopContinue);

  // Materialize passer's result for further rewriting.
  materializeValue(passerOp.getResult());

  return passerOp;
}

static LogicalResult eraseOldInit(Value oldSelector) {
  Operation *definingOp = getIndirectDefiningOp(oldSelector);
  if (auto oldInitOp = dyn_cast<InitOp>(definingOp)) {
    eraseMaterializedOperation(oldInitOp);
    return success();
  }
  return definingOp->emitError(
      "Expected the selector to be defined by an InitOp");
}

/// Returns if the value is driven by a SourceOp
static bool isSourced(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  // Heuristic to stop the traversal earlier.
  if (isa<handshake::MuxOp>(definingOp))
    return false;

  if (isa<SourceOp>(value.getDefiningOp()))
    return true;

  // If all operands of the defining operation are sourced, the value is also
  // sourced.
  return llvm::all_of(value.getDefiningOp()->getOperands(),
                      [](Value v) { return isSourced(v); });
}

/// If op is LoadOp, excludes operands coming from MemoryControllerOp.
static llvm::SmallVector<Value> getEffectiveOperands(Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For LoadOp, only the data result is effective for rewriting
    return {loadOp.getAddress()};
  }
  return llvm::to_vector(op->getOperands());
}

/// If op is LoadOp, excludes results going to MemoryControllerOp.
static llvm::SmallVector<Value> getEffectiveResults(Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For LoadOp, only the data result is effective for rewriting
    return {loadOp.getDataResult()};
  }
  // Unlike the operands, to_vector doesn't work
  llvm::SmallVector<Value> results;
  for (OpResult result : op->getResults()) {
    results.push_back(result);
  }
  return results;
}

/// Performs the motion of the OpT operation over a PM unit.
template <typename OpT>
static LogicalResult performMotion(Operation *pmOp,
                                   std::function<OpT(Value)> buildOp) {
  // Add new OpT for each effective result of the PM unit.
  for (Value result : getEffectiveResults(pmOp)) {
    OpT newOp = buildOp(result);
    inheritBB(pmOp, newOp);

    if (newOp->getNumResults() != 1)
      return newOp->emitError("Expected OpT to have a single result");

    result.replaceAllUsesExcept(newOp->getResult(0), newOp);
  }

  // Remove OpT from each effective operand of the PM unit.
  for (Value operand : getEffectiveOperands(pmOp)) {
    Operation *definingOp = operand.getDefiningOp();
    if (!isa<OpT>(definingOp)) {
      // If the operand is sourced, it doesn't need to be defined by OpT.
      if (isSourced(operand))
        continue;
      return pmOp->emitError("Expected all operands to be defined by the OpT");
    }

    if (definingOp->getNumResults() != 1)
      return pmOp->emitError("Expected OpT to have a single result");

    // The operand must be materialized to perform the motion correctly.
    assertMaterialization(operand);

    // Remove the defining OpT operation.
    definingOp->getResult(0).replaceAllUsesWith(definingOp->getOperand(0));
    definingOp->erase();
  }
  return success();
}

/// Returns if the specified PasserOp is eligible for motion past a PM unit.
static bool isEligibleForPasserMotionPastPM(PasserOp passerOp) {
  Value passerControl = passerOp.getCtrl();

  Operation *targetOp = getUniqueUser(passerOp.getResult());

  // If the targetOp is not a PM unit, return false.
  if (!isa<ArithOpInterface, NotOp, ForkOp, LazyForkOp, BufferOp, LoadOp>(
          targetOp))
    return false;

  // Iterate over operands of the targetOp to decide the eligibility for
  // motion.
  for (Value operand : getEffectiveOperands(targetOp)) {
    if (auto passerOp = dyn_cast<PasserOp>(operand.getDefiningOp())) {
      // If this passerOp is controlled by different control from the specified
      // one, not eligible.
      if (!equalsIndirectly(passerControl, passerOp.getCtrl()))
        return false;
    } else if (!isSourced(operand)) {
      // Each operand must be defined by a passer, except when it is driven by a
      // source op.
      return false;
    }
  }

  return true;
}

/// Move the specified passer past a PM unit.
static void performPasserMotionPastPM(PasserOp passerOp,
                                      DenseSet<PasserOp> &frontiers) {
  Value passerControl = passerOp.getCtrl();
  Location passerLoc = passerOp.getLoc();
  OpBuilder builder(passerOp->getContext());
  builder.setInsertionPoint(passerOp);

  Operation *targetOp = getUniqueUser(passerOp.getResult());

  // Remove passers from the frontiers
  for (Value operand : getEffectiveOperands(targetOp)) {
    if (auto passerOp = dyn_cast<PasserOp>(operand.getDefiningOp())) {
      frontiers.erase(passerOp);
    }
  }

  // Perform the motion
  auto motionResult = performMotion<PasserOp>(targetOp, [&](Value v) {
    // Use unchanged passer control.
    return builder.create<PasserOp>(passerLoc, v, passerControl);
  });

  if (failed(motionResult)) {
    targetOp->emitError("Failed to perform motion for PasserOp");
    llvm_unreachable("SpeculationV2 algorithm failed");
  }

  // Add new passers to the frontiers
  for (auto result : getEffectiveResults(targetOp)) {
    auto newPasser = cast<PasserOp>(getUniqueUser(result));
    frontiers.insert(newPasser);
    // Materialize the result of the new passer for further rewriting.
    materializeValue(newPasser.getResult());
  }
}

/// Builds an interpolator op that uses the same value for both operands.
/// Returns the interpolator op.
static SpecV2InterpolatorOp introduceIdentInterpolator(Value val) {
  OpBuilder builder(val.getContext());
  builder.setInsertionPoint(val.getDefiningOp());

  SpecV2InterpolatorOp interpolatorOp =
      builder.create<SpecV2InterpolatorOp>(val.getLoc(), val, val);
  inheritBB(val.getDefiningOp(), interpolatorOp);

  return interpolatorOp;
}

/// Adds a next interpolator for the passer induction.
static FailureOr<SpecV2InterpolatorOp>
addNextInterpolator(SpecV2InterpolatorOp oldInterpolator) {
  // The new long operand will be the result of a repeating init, which uses the
  // same value as the previous interpolator's long operand.
  Value newLongOperand = nullptr;
  for (Operation *user :
       iterateOverPossiblyIndirectUsers(oldInterpolator.getLongOperand())) {
    if (auto riOp = dyn_cast<SpecV2RepeatingInitOp>(user)) {
      // If the long operand is a SpecV2RepeatingInitOp, we can use it as the
      // new long operand
      newLongOperand = riOp.getResult();
      break;
    }
  }
  if (!newLongOperand) {
    return oldInterpolator->emitError("Expected the long operand value to be "
                                      "also used by a SpecV2RepeatingInitOp");
  }

  OpBuilder builder(oldInterpolator->getContext());
  builder.setInsertionPoint(oldInterpolator);

  // Build a new interpolator
  // Short operand remains the same
  // Long operand is the result of a repeating init
  auto newInterpolatorOp = builder.create<SpecV2InterpolatorOp>(
      oldInterpolator.getLoc(), oldInterpolator.getShortOperand(),
      newLongOperand);
  inheritBB(oldInterpolator, newInterpolatorOp);

  return newInterpolatorOp;
}

/// Returns if the circuit is eligible for the passer induction.
/// The arguments are the bottom passer and new interpolator. Other units
/// are referenced from the structure.
static bool isEligibleForPasserInduction(PasserOp bottomPasser,
                                         SpecV2InterpolatorOp newInterpolator) {
  // 1. Ensure the rewritten subcircuit structure.
  // The upstream unit of the bottom passer must be a passer.
  Operation *upstreamOp = bottomPasser.getData().getDefiningOp();
  if (!isa<PasserOp>(upstreamOp))
    return false;
  auto topPasser = cast<PasserOp>(upstreamOp);

  // 2. Ensure the context

  Operation *ctrlDefiningOp = getIndirectDefiningOp(bottomPasser.getCtrl());
  // The ctrl of the bottom passer must be generated by an interpolator.
  if (!isa<SpecV2InterpolatorOp>(ctrlDefiningOp))
    return false;
  auto oldInterpolator = cast<SpecV2InterpolatorOp>(ctrlDefiningOp);

  // The short operand must remain the same
  if (!equalsIndirectly(oldInterpolator.getShortOperand(),
                        newInterpolator.getShortOperand()))
    return false;

  // The ctrl of the top passer must be generated by a repeating init.
  Operation *topCtrlDefiningOp = getIndirectDefiningOp(topPasser.getCtrl());
  if (!isa<SpecV2RepeatingInitOp>(topCtrlDefiningOp))
    return false;
  auto topRepeatingInit = cast<SpecV2RepeatingInitOp>(topCtrlDefiningOp);

  // The top repeating init must use the same value as the old interpolator's
  // long operand.
  if (!equalsIndirectly(topRepeatingInit.getOperand(),
                        oldInterpolator.getLongOperand()))
    return false;

  // The new interpolator's long operand must be the same as the top repeating
  // init's result.
  if (!equalsIndirectly(newInterpolator.getLongOperand(),
                        topRepeatingInit.getResult()))
    return false;

  return true;
}

/// Performs the passer induction.
/// The arguments are the bottom passer and the new interpolator. Other units
/// are referenced from the structure.
static void performPasserInduction(PasserOp bottomPasser,
                                   SpecV2InterpolatorOp newInterpolator) {
  auto topPasser = cast<PasserOp>(bottomPasser.getData().getDefiningOp());

  // Perform the rewriting
  bottomPasser.getCtrlMutable()[0].set(newInterpolator.getResult());
  bottomPasser.getDataMutable()[0].set(topPasser.getData());

  // Remove the top passer
  topPasser->erase();
}

/// Move the top (least recently added) repeating init and passer below the fork
/// as the preparation for the resolver insertion.
static void moveTopRIAndPasser(SpecV2InterpolatorOp interpolator,
                               SpecV2RepeatingInitOp topRI, unsigned n) {
  auto oldPasserOp = cast<PasserOp>(topRI.getOperand().getDefiningOp());

  OpBuilder builder(topRI->getContext());

  // Materialize the result of the last repeating init for the motion.
  // When n=1, we need a nested fork structure (will be documented later).
  if (n > 1) {
    assert(!equalsIndirectly(interpolator.getShortOperand(),
                             interpolator.getLongOperand()));
    materializeValue(interpolator.getShortOperand());
  } else {
    builder.setInsertionPoint(topRI);
    // The top fork has two outputs
    auto forkOp = builder.create<ForkOp>(topRI.getLoc(), topRI.getResult(), 2);
    inheritBB(topRI, forkOp);

    // Only the interpolator uses the output#1
    interpolator.getShortOperandMutable()[0].set(forkOp.getResult()[1]);

    // Other users are assigned to the output#0, which is materialized in the
    // usual way.
    topRI.getResult().replaceAllUsesExcept(forkOp.getResult()[0], forkOp);
    materializeValue(forkOp.getResult()[0]);
  }

  // Now the user of the repeating init's result is a fork.
  auto forkOp = cast<ForkOp>(getUniqueUser(topRI.getResult()));

  // Perform repeating init motion over this fork.
  builder.setInsertionPoint(forkOp);
  if (performMotion<SpecV2RepeatingInitOp>(forkOp, [&](Value v) {
        return builder.create<SpecV2RepeatingInitOp>(topRI.getLoc(), v, 1);
      }).failed()) {
    forkOp->emitError("Failed to perform motion for SpecV2RepeatingInitOp");
    llvm_unreachable("SpeculationV2 algorithm failed");
  }

  // Perform passer motion over this fork.
  builder.setInsertionPoint(forkOp);
  if (performMotion<PasserOp>(forkOp, [&](Value v) {
        return builder.create<PasserOp>(oldPasserOp.getLoc(), v,
                                        oldPasserOp.getCtrl());
      }).failed()) {
    forkOp->emitError("Failed to perform motion for PasserOp");
    llvm_unreachable("SpeculationV2 algorithm failed");
  }
}

/// Returns if the circuit is eligible for the introduction of the resolver.
static bool
isEligibleForResolverIntroduction(SpecV2InterpolatorOp interpolator) {
  // Ensure the structure
  Operation *shortOperandDefiningOp =
      interpolator.getShortOperand().getDefiningOp();
  if (!isa<SpecV2RepeatingInitOp>(shortOperandDefiningOp))
    return false;
  auto riOp = cast<SpecV2RepeatingInitOp>(shortOperandDefiningOp);

  Operation *riOpUpstream = riOp.getOperand().getDefiningOp();
  if (!isa<PasserOp>(riOpUpstream))
    return false;

  // TODO: Confirm the longOperand
  return true;
}

/// Introduces a spec resolver.
/// Returns the resolver result value.
static Value introduceSpecResolver(SpecV2InterpolatorOp interpolator) {
  auto riOp = cast<SpecV2RepeatingInitOp>(
      (interpolator.getShortOperand().getDefiningOp()));
  auto passerOp = cast<PasserOp>(riOp.getOperand().getDefiningOp());

  OpBuilder builder(interpolator->getContext());
  builder.setInsertionPoint(interpolator);

  auto resolverOp = builder.create<SpecV2ResolverOp>(
      interpolator.getLoc(), passerOp.getData(), interpolator.getLongOperand());
  inheritBB(interpolator, resolverOp);

  interpolator.getResult().replaceAllUsesWith(resolverOp.getResult());
  interpolator->erase();
  riOp->erase();
  passerOp->erase();
  return resolverOp.getResult();
}

/// Generate Spec Loop Exit signal by building an AndIOp.
static Value generateSpecLoopExit(Value loopExit, Value confirmSpec) {
  OpBuilder builder(confirmSpec.getContext());
  builder.setInsertionPoint(confirmSpec.getDefiningOp());

  AndIOp andOp =
      builder.create<AndIOp>(confirmSpec.getLoc(), loopExit, confirmSpec);
  inheritBB(confirmSpec.getDefiningOp(), andOp);

  return andOp.getResult();
}

/// Returns if the simplification of 3 passers is possible.
static bool isPasserSimplifiable(PasserOp bottomPasser, Value cond1,
                                 Value cond2, Value andCond) {
  // Ensure the structure
  Operation *ctrlDefiningOp = bottomPasser.getCtrl().getDefiningOp();
  if (!isa<PasserOp>(ctrlDefiningOp))
    return false;
  auto ctrlDefiningPasser = cast<PasserOp>(ctrlDefiningOp);

  Operation *topOp = bottomPasser.getData().getDefiningOp();
  if (!isa<PasserOp>(topOp))
    return false;
  auto topPasser = cast<PasserOp>(topOp);

  // Confirm the context
  if (!equalsIndirectly(ctrlDefiningPasser.getData(), cond1))
    return false;

  if (!equalsIndirectly(ctrlDefiningPasser.getCtrl(), cond2))
    return false;

  if (!equalsIndirectly(topPasser.getCtrl(), cond2))
    return false;

  Operation *andCondDefiningOp = andCond.getDefiningOp();
  if (!isa<AndIOp>(andCondDefiningOp))
    return false;
  auto andOp = cast<AndIOp>(andCondDefiningOp);

  if (!equalsIndirectly(andOp.getLhs(), cond1))
    return false;
  if (!equalsIndirectly(andOp.getRhs(), cond2))
    return false;

  return true;
}

/// Simplify 3 passers into a single one.
static void simplifyPassers(PasserOp bottomPasser, Value andCond) {
  auto topPasser = cast<PasserOp>(bottomPasser.getData().getDefiningOp());
  auto ctrlDefiningPasser =
      cast<PasserOp>(bottomPasser.getCtrl().getDefiningOp());

  bottomPasser.getCtrlMutable()[0].set(andCond);
  bottomPasser.getDataMutable()[0].set(topPasser.getData());

  topPasser->erase();
  ctrlDefiningPasser->erase();
}

/// Replace the repeating init chain with a merge to enable variable
/// speculation.
/// Returns the merge.
static MergeOp replaceRIChainWithMerge(SpecV2RepeatingInitOp bottomRI,
                                       unsigned n) {

  // Find the top repeating init in the chain.
  SpecV2RepeatingInitOp topRI = bottomRI;
  for (unsigned i = 1; i < n; i++) {
    topRI = cast<SpecV2RepeatingInitOp>(topRI.getOperand().getDefiningOp());
  }

  OpBuilder builder(bottomRI->getContext());
  builder.setInsertionPoint(bottomRI);
  Location specLoc = bottomRI.getLoc();
  unsigned bb = getLogicBB(bottomRI).value();

  // Generate the source and constant providing the continue token constantly.
  SourceOp conditionGenerator = builder.create<SourceOp>(specLoc);
  setBB(conditionGenerator, bb);
  conditionGenerator->setAttr("specv2_ignore_buffer",
                              builder.getBoolAttr(true));

  ConstantOp conditionConstant = builder.create<ConstantOp>(
      specLoc, IntegerAttr::get(builder.getIntegerType(1), 1),
      conditionGenerator.getResult());
  setBB(conditionConstant, bb);
  conditionConstant->setAttr("specv2_ignore_buffer", builder.getBoolAttr(true));

  // Buffer to prevent deadlock
  // Specify "buffer_as_sink" to hide the result edge from the buffering
  // algorithm.
  BufferOp specLoopContinueTehb = builder.create<BufferOp>(
      specLoc, topRI.getOperand(), TimingInfo::break_r(), 1,
      BufferOp::ONE_SLOT_BREAK_R);
  setBB(specLoopContinueTehb, bb);
  specLoopContinueTehb->setAttr("specv2_buffer_as_sink",
                                builder.getBoolAttr(true));

  // Specify "buffer_as_source" to hide the input edges from the buffering
  // algorithm.
  MergeOp merge = builder.create<MergeOp>(
      specLoc, llvm::ArrayRef<Value>{specLoopContinueTehb.getResult(),
                                     conditionConstant.getResult()});
  setBB(merge, bb);
  merge->setAttr("specv2_buffer_as_source", builder.getBoolAttr(true));
  bottomRI.getResult().replaceAllUsesWith(merge.getResult());

  // Buffer after a merge is required, which is added in the buffering pass.

  // Remove repeating inits from the bottom of the chain.
  SpecV2RepeatingInitOp ri = bottomRI;
  SpecV2RepeatingInitOp nextRI;
  for (unsigned i = 0; i < n; i++) {
    if (i < n - 1) {
      nextRI = cast<SpecV2RepeatingInitOp>(ri.getOperand().getDefiningOp());
    }
    ri->erase();
    ri = nextRI;
  }

  return merge;
}

static FailureOr<std::pair<unsigned, unsigned>>
readFromJSON(const std::string &jsonPath) {
  // Open the speculation file
  std::ifstream inputFile(jsonPath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open kernel information file for speculation\n";
    return failure();
  }

  // Read the JSON content from the file and into a string
  std::string jsonString;
  std::string line;
  while (std::getline(inputFile, line))
    jsonString += line;

  // Try to parse the string as a JSON
  llvm::Expected<llvm::json::Value> value = llvm::json::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse kernel information file for speculation\n";
    return failure();
  }

  llvm::json::Object *jsonObject = value->getAsObject();
  if (!jsonObject) {
    llvm::errs() << "Expected a JSON object in the kernel information file for "
                    "speculation\n";
    return failure();
  }

  std::optional<int64_t> headBB = jsonObject->getInteger("spec-head-bb");
  if (!headBB) {
    llvm::errs() << "Expected 'spec-head-bb' field in the kernel information "
                    "file for speculation\n";
    return failure();
  }

  std::optional<int64_t> tailBB = jsonObject->getInteger("spec-tail-bb");
  if (!tailBB) {
    llvm::errs() << "Expected 'spec-tail-bb' field in the kernel information "
                    "file for speculation\n";
    return failure();
  }

  return std::pair<unsigned, unsigned>{static_cast<unsigned>(headBB.value()),
                                       static_cast<unsigned>(tailBB.value())};
}

void HandshakeSpeculationV2Pass::runDynamaticPass() {
  // Parse json
  auto bbOrFailure = readFromJSON(jsonPath);
  if (failed(bbOrFailure))
    return signalPassFailure();

  auto [headBB, tailBB] = bbOrFailure.value();

  ModuleOp modOp = getOperation();

  // Support only one funcOp
  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();
  OpBuilder builder(funcOp->getContext());

  // Determines whether the loop condition is inverted (i.e., the loop continues
  // when false).
  auto isInvertedOrFailure = isLoopConditionInverted(funcOp, headBB, tailBB);
  if (failed(isInvertedOrFailure))
    return signalPassFailure();

  // Replace branches with passers
  auto loopConditionsOrFailure = replaceBranchesWithPassers(funcOp, tailBB);
  if (failed(loopConditionsOrFailure))
    return signalPassFailure();
  auto [loopCondition, invertedCondition] = loopConditionsOrFailure.value();

  // Define loopContinue and loopExit based on the polarity of the condition.
  Value loopContinue, loopExit;
  if (isInvertedOrFailure.value()) {
    loopContinue = invertedCondition;
    loopExit = loopCondition;
  } else {
    loopContinue = loopCondition;
    loopExit = invertedCondition;
  }

  // Update the loop header (CMerge -> Init)
  auto selectorOrFailure =
      updateLoopHeader(funcOp, headBB, tailBB, loopContinue);
  if (failed(selectorOrFailure))
    return signalPassFailure();
  // The output of the init unit
  Value selector = selectorOrFailure.value();

  DenseSet<PasserOp> frontiers;
  SmallVector<SpecV2RepeatingInitOp> repeatingInits(n);
  Value specLoopContinue = loopContinue;
  // Repeatedly move passers past Muxes and PMSC.
  for (unsigned i = 0; i < n; i++) {
    frontiers.clear();

    // Append a repeating init and init.
    Value newSpecLoopContinue = appendRepeatingInit(specLoopContinue);
    Value newSelector = appendInit(newSpecLoopContinue);

    repeatingInits[i] =
        cast<SpecV2RepeatingInitOp>(newSpecLoopContinue.getDefiningOp());

    // Perform MuxPasserSwap for each Mux
    for (auto muxOp :
         llvm::make_early_inc_range(funcOp.getOps<handshake::MuxOp>())) {
      if (getLogicBB(muxOp) != headBB)
        continue;

      if (!isEligibleForMuxPasserSwap(muxOp, newSelector,
                                      newSpecLoopContinue)) {
        muxOp.emitWarning("MuxOp is not eligible for Passer swap, skipping");
        continue;
      }

      auto passerOp =
          performMuxPasserSwap(muxOp, newSelector, newSpecLoopContinue);

      // Add the moved passer to the frontiers
      frontiers.insert(passerOp);
    }

    // After MuxPasserSwap, remove the old init op.
    if (failed(eraseOldInit(selector)))
      return signalPassFailure();

    // Repeatedly move passers inside PMSC.
    bool frontiersUpdated;
    do {
      frontiersUpdated = false;
      for (auto passerOp : frontiers) {
        if (isEligibleForPasserMotionPastPM(passerOp)) {
          performPasserMotionPastPM(passerOp, frontiers);
          frontiersUpdated = true;
          // If frontiers are updated, the iterator is outdated.
          // Break and restart the loop.
          break;
        }
      }
      // If no frontiers were updated, we can stop.
    } while (frontiersUpdated);

    // Update the values and repeat the iteration
    specLoopContinue = newSpecLoopContinue;
    selector = newSelector;
  }

  if (n > 0) {
    // Reduce the passer chain by introducing interpolator op and performing
    // induction.

    // Introduce a trivial interpolator
    SpecV2InterpolatorOp interpolator =
        introduceIdentInterpolator(repeatingInits[0].getResult());

    /// The interpolator is used exclusively by PasserOps during chain
    /// reduction.
    /// Passers using the oldest Spec Loop Continue must qualify for passer
    /// induction.
    repeatingInits[0].getResult().replaceUsesWithIf(
        interpolator.getResult(),
        [](OpOperand &operand) { return isa<PasserOp>(operand.getOwner()); });

    // Perform induction
    for (unsigned i = 1; i < n; i++) {
      // Add a new interpolator
      auto newInterpolatorOrFailure = addNextInterpolator(interpolator);
      if (failed(newInterpolatorOrFailure))
        return signalPassFailure();
      SpecV2InterpolatorOp newInterpolator = newInterpolatorOrFailure.value();

      // Perform the induction for each passer
      for (Operation *user :
           llvm::make_early_inc_range(interpolator.getResult().getUsers())) {
        if (auto passer = dyn_cast<PasserOp>(user)) {
          if (isEligibleForPasserInduction(passer, newInterpolator)) {
            performPasserInduction(passer, newInterpolator);
          } else {
            passer->emitError("The passer is not eligible for induction");
            return signalPassFailure();
          }
        }
      }

      if (!interpolator.getResult().use_empty()) {
        interpolator.emitError("The old interpolator still has users.");
        return signalPassFailure();
      }

      // Erase the old interpolator
      interpolator->erase();
      interpolator = newInterpolator;
    }

    // Preparation for the resolver insertion
    moveTopRIAndPasser(interpolator, /*topRI=*/repeatingInits[0], n);

    // Introduce the resolver
    if (!isEligibleForResolverIntroduction(interpolator)) {
      interpolator.emitError(
          "The circuit is not eligible for the resolver introduction");
      return signalPassFailure();
    }
    Value confirmSpec = introduceSpecResolver(interpolator);

    // Simplify the exit passers.
    Value specLoopExit = generateSpecLoopExit(loopExit, confirmSpec);
    for (Operation *user : iterateOverPossiblyIndirectUsers(loopExit)) {
      if (auto topPasser = dyn_cast<PasserOp>(user)) {
        if (equalsIndirectly(topPasser.getCtrl(), confirmSpec)) {
          // Passer's result is materialized
          Operation *downstreamOp = getUniqueUser(topPasser.getResult());
          if (auto bottomPasser = dyn_cast<PasserOp>(downstreamOp)) {
            if (isPasserSimplifiable(bottomPasser, loopExit, confirmSpec,
                                     specLoopExit)) {
              simplifyPassers(bottomPasser, specLoopExit);
            } else {
              bottomPasser->emitError(
                  "Expected the exit passer to be simplifiable");
            }
          }
        }
      }
    }

    if (variable) {
      MergeOp merge = replaceRIChainWithMerge(repeatingInits[n - 1], n);

      // Optimize for buffering
      auto passer = cast<PasserOp>(
          merge->getOperand(0).getDefiningOp()->getOperand(0).getDefiningOp());
      passer.getCtrlMutable()[0].set(specLoopExit);
    }
  }

  // Erase unused PasserOps
  for (auto passerOp : llvm::make_early_inc_range(funcOp.getOps<PasserOp>())) {
    Value result = passerOp.getResult();

    if (result.use_empty())
      passerOp->erase();
    else if (result.hasOneUse()) {
      Operation *user = getUniqueUser(result);
      if (isa<SinkOp>(user)) {
        user->erase();
        passerOp->erase();
      }
    }
  }
}
