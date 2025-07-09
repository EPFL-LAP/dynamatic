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

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculationv2;

namespace {

struct HandshakeSpeculationV2Pass
    : public dynamatic::experimental::speculationv2::impl::
          HandshakeSpeculationV2Base<HandshakeSpeculationV2Pass> {
  HandshakeSpeculationV2Pass() {}

  void runDynamaticPass() override;
};
} // namespace

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
/// Can be performed to branches inside a loop, or to branches not at the bottom
/// of the loop (when there are multiple loop exits).
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

    // The condition must be the same
    if (!equalsForContext(condition, branch.getConditionOperand()))
      return branch.emitError("Branch condition does not match the condition "
                              "of the reference branch");

    Value data = branch.getDataOperand();

    builder.setInsertionPoint(branch);

    // Build a passer for the trueResult, unless the user is a sink.
    Operation *trueResultUser = getUniqueUser(branch.getTrueResult());
    if (isa<SinkOp>(trueResultUser)) {
      trueResultUser->erase();
    } else {
      PasserOp passer =
          builder.create<PasserOp>(branch.getLoc(), data, condition);
      setBB(passer, bb);
      branch.getTrueResult().replaceAllUsesWith(passer.getResult());
    }

    // Build a passer for the falseResult, unless the user is a sink.
    // The passer ctrl is inverted condition.
    Operation *falseResultUser = getUniqueUser(branch.getFalseResult());
    if (isa<SinkOp>(falseResultUser)) {
      falseResultUser->erase();
    } else {
      PasserOp passer = builder.create<PasserOp>(branch.getLoc(), data,
                                                 invertCondition.getResult());
      setBB(passer, bb);
      branch.getFalseResult().replaceAllUsesWith(passer.getResult());
    }

    branch->erase();
  }

  return std::pair<Value, Value>{condition, invertCondition.getResult()};
}

/// Replace the CMerge-controlled loop header with InitOp[False].
static FailureOr<Value> updateLoopHeader(FuncOp &funcOp, unsigned loopHeadBB,
                                         unsigned loopTailBB,
                                         Value loopContinue) {
  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(funcOp.getBodyBlock(),
                            funcOp.getBodyBlock()->begin());

  // Build an InitOp[False]
  InitOp initOp = builder.create<InitOp>(loopContinue.getLoc(), loopContinue);
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

  // Inverted: the backedge is the first operand.
  bool isInverted;
  Operation *definingOp0 = cmergeOp.getDataOperands()[0].getDefiningOp();
  Operation *definingOp1 = cmergeOp.getDataOperands()[1].getDefiningOp();
  Value entry, backedge;
  if (definingOp0 && getLogicBB(definingOp0) == loopTailBB) {
    isInverted = true;
    entry = cmergeOp.getDataOperands()[1];
    backedge = cmergeOp.getDataOperands()[0];
  } else if (definingOp1 && getLogicBB(definingOp1) == loopTailBB) {
    isInverted = false;
    entry = cmergeOp.getDataOperands()[0];
    backedge = cmergeOp.getDataOperands()[1];
  }

  // Build a MuxOp to replace the CMergeOp
  builder.setInsertionPoint(cmergeOp);
  MuxOp muxOp =
      builder.create<MuxOp>(cmergeOp.getLoc(), cmergeOp.getResult().getType(),
                            loopContinue, llvm::ArrayRef{entry, backedge});
  setBB(muxOp, loopHeadBB);
  cmergeOp.getResult().replaceAllUsesWith(muxOp.getResult());

  // Update muxes
  for (auto muxOp : funcOp.getOps<handshake::MuxOp>()) {
    if (getLogicBB(muxOp) != loopHeadBB)
      continue;

    assert(muxOp.getDataOperands().size() == 2);

    // Update the select operand
    muxOp.getSelectOperandMutable()[0].set(initOp.getResult());

    if (isInverted) {
      // Swap operands
      Value entry = muxOp.getDataOperands()[1];
      muxOp.getDataOperandsMutable()[1].set(muxOp.getDataOperands()[0]);
      muxOp.getDataOperandsMutable()[0].set(entry);
    }
  }

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
      builder.create<SpecV2RepeatingInitOp>(val.getLoc(), val);
  inheritBB(val.getDefiningOp(), repeatingInitOp);

  return repeatingInitOp.getResult();
}

/// Appends an init to the value. Returns the result of the init.
static Value appendInit(Value val) {
  OpBuilder builder(val.getContext());
  builder.setInsertionPoint(val.getDefiningOp());

  InitOp initOp = builder.create<InitOp>(val.getLoc(), val);
  inheritBB(val.getDefiningOp(), initOp);

  return initOp.getResult();
}

/// Returns if the MuxPasserSwap is eligible.
static bool isMuxPasserSwapEligible(MuxOp muxOp, Value newSelector,
                                    Value newSpecLoopContinue) {
  // Ensure the rewritten subcircuit structure.
  Operation *backedgeDefiningOp = muxOp.getDataOperands()[1].getDefiningOp();
  if (!isa<PasserOp>(backedgeDefiningOp))
    return false;
  auto passerOp = cast<PasserOp>(backedgeDefiningOp);

  // Ensure the context
  Operation *selectorDefiningOp =
      getDefiningOpForContext(muxOp.getSelectOperand());
  if (!isa<InitOp>(selectorDefiningOp))
    return false;
  auto initOp = cast<InitOp>(selectorDefiningOp);

  if (!equalsForContext(passerOp.getCtrl(), initOp.getOperand()))
    return false;

  Operation *newSelectorDefiningOp = getDefiningOpForContext(newSelector);
  if (!isa<InitOp>(newSelectorDefiningOp))
    return false;
  auto newInitOp = cast<InitOp>(newSelectorDefiningOp);

  if (!equalsForContext(newInitOp.getOperand(), newSpecLoopContinue))
    return false;

  Operation *newSpecLoopContinueDefiningOp =
      getDefiningOpForContext(newSpecLoopContinue);
  if (!isa<SpecV2RepeatingInitOp>(newSpecLoopContinueDefiningOp))
    return false;
  auto newRepeatingInitOp =
      cast<SpecV2RepeatingInitOp>(newSpecLoopContinueDefiningOp);
  if (!equalsForContext(newRepeatingInitOp.getOperand(), initOp.getOperand()))
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
  Operation *definingOp = getDefiningOpForContext(oldSelector);
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
  return llvm::all_of(value.getDefiningOp()->getOperands(),
                      [](Value v) { return isSourced(v); });
}

/// Returns the operands excluding the channels to MemoryControllerOp for
/// LoadOp.
static llvm::SmallVector<Value> getSubjectOperands(Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For LoadOp, only the data result is considered a subject operand
    return {loadOp.getAddress()};
  }
  return llvm::to_vector(op->getOperands());
}

/// Returns the results excluding the channels to MemoryControllerOp for
/// LoadOp.
static llvm::SmallVector<Value> getSubjectResults(Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For LoadOp, only the data result is considered a subject operand
    return {loadOp.getDataResult()};
  }
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
  for (Value result : getSubjectResults(pmOp)) {
    OpT newOp = buildOp(result);
    inheritBB(pmOp, newOp);

    if (newOp->getNumResults() != 1)
      return newOp->emitError("Expected OpT to have a single result");

    result.replaceAllUsesExcept(newOp->getResult(0), newOp);
  }

  for (Value operand : getSubjectOperands(pmOp)) {
    if (isSourced(operand))
      continue;

    // The operand must be materialized to perform the motion correctly.
    assertMaterialization(operand);

    Operation *definingOp = operand.getDefiningOp();
    if (!isa<OpT>(definingOp))
      return pmOp->emitError("Expected all operands to be defined by the OpT");
    if (definingOp->getNumResults() != 1)
      return pmOp->emitError("Expected OpT to have a single result");

    definingOp->getResult(0).replaceAllUsesWith(definingOp->getOperand(0));
    definingOp->erase();
  }
  return success();
}

/// Tries moving the specified passer past a PM unit.
/// Returns true if the motion was successful, false otherwise.
static bool tryMovingPasser(PasserOp passerOp, DenseSet<PasserOp> &frontiers) {
  Value passerControl = passerOp.getCtrl();
  Location passerLoc = passerOp.getLoc();
  OpBuilder builder(passerOp->getContext());
  builder.setInsertionPoint(passerOp);

  Operation *targetOp = getUniqueUser(passerOp.getResult());

  if (isa<ArithOpInterface, NotOp, ForkOp, LazyForkOp, BufferOp, LoadOp>(
          targetOp)) {

    DenseSet<PasserOp> passersToBeMoved;

    bool isEligible = true;

    // Iterate over operands of the targetOp to decide the eligibility for
    // motion. getSubjectOperands excludes channels connected to
    // MemoryControllerOp for LoadOp.
    for (Value operand : getSubjectOperands(targetOp)) {
      if (auto passerOp = dyn_cast<PasserOp>(operand.getDefiningOp())) {
        // If another passerOp is controlled by different control, not eligible.
        if (!equalsForContext(passerControl, passerOp.getCtrl())) {
          isEligible = false;
          break;
        }
        assertMaterialization(operand);
        passersToBeMoved.insert(passerOp);
        continue;
      }

      // Even if the operand is not defined by a PasserOp, if it is sourced,
      // eligible.
      if (!isSourced(operand)) {
        isEligible = false;
        break;
      }
    }

    if (isEligible) {
      // Erase the passers that are going to be moved
      for (auto passer : passersToBeMoved)
        frontiers.erase(passer);

      // Perform the motion
      auto motionResult = performMotion<PasserOp>(targetOp, [&](Value v) {
        return builder.create<PasserOp>(passerLoc, v, passerControl);
      });

      if (failed(motionResult)) {
        targetOp->emitError("Failed to perform motion for PasserOp");
        llvm_unreachable("SpeculationV2 algorithm failed");
      }

      // Add the new passer to the frontiers
      for (auto result : getSubjectResults(targetOp)) {
        auto newPasser = cast<PasserOp>(getUniqueUser(result));
        frontiers.insert(newPasser);
        // Materialize the result of the new passer for further rewriting.
        materializeValue(newPasser.getResult());
      }

      // Motion is performed
      return true;
    }
  }
  return false;
}

/// Builds an interpolator op that uses the same value for both operands.
/// Returns the interpolator op.
static SpecV2InterpolatorOp introduceIdentInterpolator(Value val) {
  OpBuilder builder(val.getContext());
  builder.setInsertionPoint(val.getDefiningOp());

  SpecV2InterpolatorOp interpolatorOp =
      builder.create<SpecV2InterpolatorOp>(val.getLoc(), val, val);
  inheritBB(val.getDefiningOp(), interpolatorOp);

  /// Interpolator is only used by PasserOps for the chain reduction.
  /// Passers using the oldest Spec Loop Continue must be eligible for the
  /// passer induction.
  val.replaceUsesWithIf(interpolatorOp.getResult(), [](OpOperand &operand) {
    return isa<PasserOp>(operand.getOwner());
  });

  return interpolatorOp;
}

static FailureOr<SpecV2InterpolatorOp>
addNextInterpolator(SpecV2InterpolatorOp interpolatorOp) {
  // The new long operand will be the result of a repeating init, which uses the
  // same value as the previous interpolator's long operand.
  Value newLongOperand = nullptr;
  for (Operation *user :
       iterateOverPossiblyMaterializedUsers(interpolatorOp.getLongOperand())) {
    if (auto riOp = dyn_cast<SpecV2RepeatingInitOp>(user)) {
      // If the long operand is a SpecV2RepeatingInitOp, we can use it as the
      // new long operand
      newLongOperand = riOp.getResult();
      break;
    }
  }
  if (!newLongOperand) {
    return interpolatorOp->emitError("Expected the long operand value to be "
                                     "also used by a SpecV2RepeatingInitOp");
  }

  OpBuilder builder(interpolatorOp->getContext());
  builder.setInsertionPoint(interpolatorOp);

  // Build a new interpolator
  // Short operand remains the same
  // Long operand is the result of a repeating init
  auto newInterpolatorOp = builder.create<SpecV2InterpolatorOp>(
      interpolatorOp.getLoc(), interpolatorOp.getShortOperand(),
      newLongOperand);
  inheritBB(interpolatorOp, newInterpolatorOp);

  return newInterpolatorOp;
}

/// Returns if the passer induction is eligible.
/// The arguments are the bottom passer and the new interpolator. Other units
/// are referenced from the structure.
static bool isPasserInductionEligible(PasserOp bottomPasser,
                                      SpecV2InterpolatorOp newInterpolator) {
  // 1. Ensure the rewritten subcircuit structure.
  // The upstream unit of the bottom passer must be a passer.
  Operation *upstreamOp = bottomPasser.getData().getDefiningOp();
  if (!isa<PasserOp>(upstreamOp))
    return false;
  auto topPasser = cast<PasserOp>(upstreamOp);

  // 2. Ensure the context

  Operation *ctrlDefiningOp = getDefiningOpForContext(bottomPasser.getCtrl());
  // The ctrl of the bottom passer must be generated by an interpolator.
  if (!isa<SpecV2InterpolatorOp>(ctrlDefiningOp))
    return false;

  // The short operand must remain the same
  auto oldInterpolator = cast<SpecV2InterpolatorOp>(ctrlDefiningOp);
  if (!equalsForContext(oldInterpolator.getShortOperand(),
                        newInterpolator.getShortOperand()))
    return false;

  // The ctrl of the top passer must be generated by a repeating init.
  Operation *topCtrlDefiningOp = getDefiningOpForContext(topPasser.getCtrl());
  if (!isa<SpecV2RepeatingInitOp>(topCtrlDefiningOp))
    return false;

  // The top repeating init must use the same value as the old interpolator's
  // long operand.
  auto topRepeatingInit = cast<SpecV2RepeatingInitOp>(topCtrlDefiningOp);
  if (!equalsForContext(topRepeatingInit.getOperand(),
                        oldInterpolator.getLongOperand()))
    return false;

  // The new interpolator's long operand must be the same as the top repeating
  // init's result.
  if (!equalsForContext(newInterpolator.getLongOperand(),
                        topRepeatingInit.getResult()))
    return false;

  return true;
}

/// Performs the passer induction.
/// The arguments are the bottom passer and the new interpolator. Other units
/// are referenced from the structure.
static void runPasserInduction(PasserOp bottomPasser,
                               SpecV2InterpolatorOp newInterpolator) {
  auto topPasser = cast<PasserOp>(bottomPasser.getData().getDefiningOp());

  // Perform the rewriting
  bottomPasser.getCtrlMutable()[0].set(newInterpolator.getResult());
  bottomPasser.getDataMutable()[0].set(topPasser.getData());

  // Remove the top passer
  topPasser->erase();
}

/// Move the top (least recently added) repeating init and passer down the fork
/// as a preparation for the resolver insertion.
static void moveTopRIAndPasser(SpecV2InterpolatorOp interpolator, unsigned n) {
  // TODO: update the argument

  auto oldRepeatingInitOp = cast<SpecV2RepeatingInitOp>(
      getDefiningOpForContext(interpolator.getShortOperand()));
  auto oldPasserOp =
      cast<PasserOp>(getDefiningOpForContext(oldRepeatingInitOp.getOperand()));

  OpBuilder builder(interpolator->getContext());

  // Materialize the result of the last repeating init.
  // When n=1, we need a nested fork for the appropriate motion of repeating
  // init and passer later.
  if (n > 1) {
    assert(!equalsForContext(interpolator.getShortOperand(),
                             interpolator.getLongOperand()));
    materializeValue(interpolator.getShortOperand());
  } else {
    builder.setInsertionPoint(oldRepeatingInitOp);
    // The top fork has two outputs
    auto forkOp = builder.create<ForkOp>(oldRepeatingInitOp.getLoc(),
                                         oldRepeatingInitOp.getResult(), 2);
    inheritBB(oldRepeatingInitOp, forkOp);

    // Only the interpolator uses the output#1
    interpolator.getShortOperandMutable()[0].set(forkOp.getResult()[1]);

    // Other users are allocated to the output#0, which is materialized in a
    // usual way.
    oldRepeatingInitOp.getResult().replaceAllUsesExcept(forkOp.getResult()[0],
                                                        forkOp);
    materializeValue(forkOp.getResult()[0]);
  }

  // Now the user of the repeating init's result is a fork.
  auto forkOp = cast<ForkOp>(getUniqueUser(oldRepeatingInitOp.getResult()));

  // Perform repeating init motion
  builder.setInsertionPoint(forkOp);
  if (performMotion<SpecV2RepeatingInitOp>(forkOp, [&](Value v) {
        return builder.create<SpecV2RepeatingInitOp>(
            oldRepeatingInitOp.getLoc(), v);
      }).failed()) {
    forkOp->emitError("Failed to perform motion for SpecV2RepeatingInitOp");
    llvm_unreachable("SpeculationV2 algorithm failed");
  }

  // Perform passer motion
  builder.setInsertionPoint(forkOp);
  if (performMotion<PasserOp>(forkOp, [&](Value v) {
        return builder.create<PasserOp>(oldPasserOp.getLoc(), v,
                                        oldPasserOp.getCtrl());
      }).failed()) {
    forkOp->emitError("Failed to perform motion for PasserOp");
    llvm_unreachable("SpeculationV2 algorithm failed");
  }
}

/// Returns if the introduction of the resolver is eligible.
static bool
isIntroductionOfResolverEligible(SpecV2InterpolatorOp interpolator) {
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

/// Introduce a spec resolver.
static Value introduceSpecResolver(SpecV2InterpolatorOp interpolator) {
  auto riOp = cast<SpecV2RepeatingInitOp>(
      getDefiningOpForContext(interpolator.getShortOperand()));
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
      builder.create<AndIOp>(confirmSpec.getLoc(), confirmSpec, loopExit);
  inheritBB(confirmSpec.getDefiningOp(), andOp);

  return andOp.getResult();
}

static bool isExitPasserSimplifiable(PasserOp bottomPasser, Value loopExit,
                                     Value confirmSpec) {
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
  if (!equalsForContext(ctrlDefiningPasser.getData(), loopExit))
    return false;

  if (!equalsForContext(ctrlDefiningPasser.getCtrl(), confirmSpec))
    return false;

  return equalsForContext(topPasser.getCtrl(), confirmSpec);
}

static void simplifyExitPasser(PasserOp bottomPasser, Value specLoopExit) {
  auto topPasser = cast<PasserOp>(bottomPasser.getData().getDefiningOp());
  bottomPasser.getCtrlMutable()[0].set(specLoopExit);
  bottomPasser.getDataMutable()[0].set(topPasser.getData());
  topPasser->erase();
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

void HandshakeSpeculationV2Pass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();
  OpBuilder builder(funcOp->getContext());

  // Obtain if the loop condition is inverted (i.e., false continues the loop).
  auto isNegatedOrFailure = isLoopConditionInverted(funcOp, headBB, tailBB);
  if (failed(isNegatedOrFailure))
    return signalPassFailure();

  // Replace branches with passers
  auto loopConditionsOrFailure = replaceBranchesWithPassers(funcOp, tailBB);
  if (failed(loopConditionsOrFailure))
    return signalPassFailure();
  auto [loopCondition, invertedCondition] = loopConditionsOrFailure.value();

  // Define loopContinue and loopExit based on the negation of the condition.
  Value loopContinue, loopExit;
  if (isNegatedOrFailure.value()) {
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
  SmallVector<Value> specLoopContinues(n);
  Value specLoopContinue = loopContinue;
  // Repeatedly move passers past Muxes and PMSC.
  for (unsigned i = 0; i < n; i++) {
    frontiers.clear();

    // Append a repeating init and init before MuxPasserSwap.
    Value newSpecLoopContinue = appendRepeatingInit(specLoopContinue);
    Value newSelector = appendInit(newSpecLoopContinue);

    specLoopContinues[i] = newSpecLoopContinue;

    // Perform MuxPasserSwap for each Mux
    for (auto muxOp :
         llvm::make_early_inc_range(funcOp.getOps<handshake::MuxOp>())) {
      if (getLogicBB(muxOp) != headBB)
        continue;

      if (!isMuxPasserSwapEligible(muxOp, newSelector, newSpecLoopContinue)) {
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
        // Try to move the passerOp with other passers in the frontiers.
        if (tryMovingPasser(passerOp, frontiers)) {
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
    // Reduce the passer chain by introducing interpolator op and reduction.

    // Introduce a trivial interpolator
    SpecV2InterpolatorOp interpolator =
        introduceIdentInterpolator(specLoopContinues[0]);

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
          if (isPasserInductionEligible(passer, newInterpolator)) {
            runPasserInduction(passer, newInterpolator);
          }
          // Some passers (e.g., passers on the backedges) are not eligible, and
          // just ignore them.
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
    moveTopRIAndPasser(interpolator, n);

    // Introduce the resolver
    if (!isIntroductionOfResolverEligible(interpolator)) {
      interpolator.emitError(
          "The introduction of the resolver is not eligible");
      return signalPassFailure();
    }
    Value confirmSpec = introduceSpecResolver(interpolator);

    Value specLoopExit = generateSpecLoopExit(loopExit, confirmSpec);
    // Simplify the exit passers.
    for (Operation *user : iterateOverPossiblyMaterializedUsers(loopExit)) {
      if (auto topPasser = dyn_cast<PasserOp>(user)) {
        if (topPasser.getCtrl() == confirmSpec) {
          // Passer's result must be materialized
          Operation *downstreamOp = getUniqueUser(topPasser.getResult());
          if (auto bottomPasser = dyn_cast<PasserOp>(downstreamOp)) {
            if (isExitPasserSimplifiable(bottomPasser, loopExit, confirmSpec)) {
              simplifyExitPasser(bottomPasser, specLoopExit);
            } else {
              bottomPasser->emitError(
                  "Expected the exit passer to be unifiable");
            }
          }
        }
      }
    }

    if (variable) {
      MergeOp merge = replaceRIChainWithMerge(
          cast<SpecV2RepeatingInitOp>(specLoopContinues[n - 1].getDefiningOp()),
          n);

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

    if (result.hasOneUse()) {
      Operation *user = getUniqueUser(result);
      if (isa<SinkOp>(user)) {
        user->erase();
        passerOp->erase();
      }
    }
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculationv2::createHandshakeSpeculationV2() {
  return std::make_unique<HandshakeSpeculationV2Pass>();
}
