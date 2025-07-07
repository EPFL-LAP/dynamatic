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

#include "experimental/Transforms/SpeculationV2/HandshakeSpeculationV2.h"
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
#include "llvm/ADT/TypeSwitch.h"
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

static FailureOr<std::pair<Value, bool>>
findLoopCondition(FuncOp &funcOp, unsigned loopHeadBB, unsigned loopTailBB) {
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
      getLogicBB(condBrOp.getTrueResult().getDefiningOp());
  std::optional<unsigned> falseResultBB =
      getLogicBB(condBrOp.getTrueResult().getDefiningOp());
  if (trueResultBB && *trueResultBB == loopHeadBB) {
    // The loop continue is the condition operand
    return std::pair<Value, bool>{condBrOp.getConditionOperand(), false};
  }
  if (falseResultBB && *falseResultBB == loopHeadBB) {
    // The loop continue is the inverted condition operand
    return std::pair<Value, bool>{condBrOp.getConditionOperand(), true};
  }
  return funcOp.emitError("Either true or false result of ConditionalBranchOp "
                          "is not the loop backedge.");
}

static std::pair<Value, Value> generateLoopContinueAndExit(OpBuilder &builder,
                                                           Value loopCondition,
                                                           bool isInverted) {
  builder.setInsertionPoint(loopCondition.getDefiningOp());
  NotOp invertCondition =
      builder.create<NotOp>(loopCondition.getLoc(), loopCondition);
  inheritBB(loopCondition.getDefiningOp(), invertCondition);
  if (isInverted)
    return {invertCondition.getResult(), loopCondition};
  else
    return {loopCondition, invertCondition.getResult()};
}

static void replaceBranches(FuncOp &funcOp, unsigned loopTailBB,
                            bool isInverted, Value loopContinue,
                            Value loopExit) {
  OpBuilder builder(funcOp->getContext());
  Value loopConditionTrue, loopConditionFalse;
  if (isInverted) {
    loopConditionTrue = loopExit;
    loopConditionFalse = loopContinue;
  } else {
    loopConditionTrue = loopContinue;
    loopConditionFalse = loopExit;
  }

  // Replace all branches in the specBB with the speculator
  for (auto branchOp :
       llvm::make_early_inc_range(funcOp.getOps<ConditionalBranchOp>())) {
    if (getLogicBB(branchOp) != loopTailBB)
      continue;

    builder.setInsertionPoint(branchOp);

    Operation *trueResultUser = *branchOp.getTrueResult().getUsers().begin();
    if (isa<SinkOp>(trueResultUser)) {
      trueResultUser->erase();
    } else {
      PasserOp loopSuppressor = builder.create<PasserOp>(
          branchOp.getLoc(), branchOp.getDataOperand(), loopConditionTrue);
      inheritBB(branchOp, loopSuppressor);
      branchOp.getTrueResult().replaceAllUsesWith(loopSuppressor.getResult());
    }

    Operation *falseResultUser = *branchOp.getFalseResult().getUsers().begin();
    if (isa<SinkOp>(falseResultUser)) {
      falseResultUser->erase();
    } else {
      PasserOp exitSuppressor = builder.create<PasserOp>(
          branchOp.getLoc(), branchOp.getDataOperand(), loopConditionFalse);
      inheritBB(branchOp, exitSuppressor);
      branchOp.getFalseResult().replaceAllUsesWith(exitSuppressor.getResult());
    }

    branchOp->erase();
  }
}

static Value replaceLoopHeaders(FuncOp &funcOp, unsigned loopHeadBB,
                                unsigned loopTailBB, Value loopContinue) {
  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(funcOp.getBodyBlock(),
                            funcOp.getBodyBlock()->begin());

  InitOp initOp = builder.create<InitOp>(loopContinue.getLoc(), loopContinue);
  setBB(initOp, loopHeadBB);

  for (auto muxOp : funcOp.getOps<handshake::MuxOp>()) {
    if (getLogicBB(muxOp) != loopHeadBB)
      continue;

    assert(muxOp.getDataOperands().size() == 2 &&
           "MuxOp in specBB should have two data operands");

    muxOp.getSelectOperandMutable()[0].set(initOp.getResult());

    Operation *definingOp = muxOp.getDataOperands()[1].getDefiningOp();
    if (!definingOp || getLogicBB(definingOp) != loopTailBB) {
      // Backedge is the second operand, so swap operands
      Value entry = muxOp.getDataOperands()[1];
      muxOp.getDataOperandsMutable()[1].set(muxOp.getDataOperands()[0]);
      muxOp.getDataOperandsMutable()[0].set(entry);
    }
  }

  for (auto cmergeOp :
       llvm::make_early_inc_range(funcOp.getOps<handshake::ControlMergeOp>())) {
    if (getLogicBB(cmergeOp) != loopHeadBB)
      continue;

    assert(cmergeOp.getDataOperands().size() == 2 &&
           "ControlMergeOp in specBB should have two data operands");

    Value entry, backedge;
    Operation *definingOp = cmergeOp.getDataOperands()[1].getDefiningOp();
    if (!definingOp || getLogicBB(definingOp) != loopTailBB) {
      entry = cmergeOp.getDataOperands()[1];
      backedge = cmergeOp.getDataOperands()[0];
    } else {
      entry = cmergeOp.getDataOperands()[0];
      backedge = cmergeOp.getDataOperands()[1];
    }

    builder.setInsertionPoint(cmergeOp);
    MuxOp muxOp = builder.create<MuxOp>(cmergeOp.getLoc(), backedge.getType(),
                                        initOp.getResult(),
                                        llvm::ArrayRef{entry, backedge});
    inheritBB(cmergeOp, muxOp);

    cmergeOp.getResult().replaceAllUsesWith(muxOp.getResult());

    // Erase the old fork
    (*cmergeOp.getIndex().getUsers().begin())->erase();

    cmergeOp->erase();
  }

  return initOp.getResult();
}

static Value appendRepeatingInit(OpBuilder &builder, Value specLoopContinue) {
  builder.setInsertionPoint(specLoopContinue.getDefiningOp());
  SpecV2RepeatingInitOp repeatingInitOp = builder.create<SpecV2RepeatingInitOp>(
      specLoopContinue.getLoc(), specLoopContinue);
  inheritBB(specLoopContinue.getDefiningOp(), repeatingInitOp);
  return repeatingInitOp.getResult();
}

static Value appendInit(OpBuilder &builder, Value specLoopContinue) {
  builder.setInsertionPoint(specLoopContinue.getDefiningOp());
  InitOp initOp =
      builder.create<InitOp>(specLoopContinue.getLoc(), specLoopContinue);
  inheritBB(specLoopContinue.getDefiningOp(), initOp);
  return initOp.getResult();
}

static void assertResultIsMaterialized(PasserOp passerOp) {
  Value result = passerOp.getResult();
  if (result.use_empty())
    return;
  if (result.hasOneUse())
    return;
  passerOp->emitError("PasserOp has multiple users, and the algorithm cannot "
                      "proceed. When PasserOp is placed, its result needs to "
                      "be materializeits result needs to be materialized.");
  llvm_unreachable("SpeculationV2 algorithm failed");
}

static void materializeValue(Value val) {
  if (val.use_empty())
    return;
  if (val.hasOneUse())
    return;

  unsigned numUses = std::distance(val.getUses().begin(), val.getUses().end());

  Operation *definingOp = val.getDefiningOp();
  OpBuilder builder(definingOp->getContext());
  builder.setInsertionPoint(definingOp);
  ForkOp forkOp = builder.create<ForkOp>(val.getLoc(), val, numUses);
  inheritBB(definingOp, forkOp);

  int i = 0;
  // To allow the mutation of operands, we use early increment range
  // TODO: Maybe he was not aware of this approach and the materialization pass
  // is dirty. Update it to use early increment range as well.
  for (OpOperand &opOperand : llvm::make_early_inc_range(val.getUses())) {
    if (opOperand.getOwner() == forkOp)
      continue;
    opOperand.set(forkOp.getResult()[i]);
    i++;
  }
}

static llvm::SmallVector<Operation *>
iterateOverPossiblyMaterializedUsers(Value result) {
  llvm::SmallVector<Operation *> users;
  for (Operation *user : result.getUsers()) {
    if (auto forkUser = dyn_cast<ForkOp>(user)) {
      // If the user is a ForkOp, we need to iterate over its results
      for (Value forkResult : forkUser.getResults()) {
        if (forkResult.hasOneUse()) {
          users.push_back(*forkResult.getUsers().begin());
        } else {
          forkUser->emitError("Expected the fork to be materialized.");
          llvm_unreachable("SpeculationV2 algorithm failed");
        }
      }
    } else {
      users.push_back(user);
    }
  }
  return users;
}

static bool isEqualUnderMaterialization(Value a, Value b) {
  if (auto fork = dyn_cast<ForkOp>(a.getDefiningOp())) {
    a = fork.getOperand();
  }
  if (auto fork = dyn_cast<ForkOp>(b.getDefiningOp())) {
    b = fork.getOperand();
  }
  return a == b;
}

static FailureOr<PasserOp> performMuxSupRewriting(MuxOp muxOp,
                                                  Value newSelector,
                                                  Value newSpecLoopContinue) {
  if (auto passerOp =
          dyn_cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp())) {
    assertResultIsMaterialized(passerOp);
    muxOp.getDataOperandsMutable()[1].set(passerOp.getData());
    passerOp.getDataMutable()[0].set(muxOp.getResult());
    muxOp.getResult().replaceAllUsesExcept(passerOp.getResult(), passerOp);
    muxOp.getSelectOperandMutable()[0].set(newSelector);
    passerOp.getCtrlMutable()[0].set(newSpecLoopContinue);

    materializeValue(passerOp.getResult());

    return passerOp;
  }
  return muxOp.emitError(
      "Expected the first data operand of MuxOp to come from a PasserOp");
}

static LogicalResult eraseOldInit(Value oldSelector) {
  if (auto oldInitOp = dyn_cast<InitOp>(oldSelector.getDefiningOp())) {
    oldInitOp->erase();
    return success();
  }
  return oldSelector.getDefiningOp()->emitError(
      "Expected the selector to be defined by an InitOp");
}

static bool isSourced(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  // Heuristic
  if (isa<handshake::MuxOp>(definingOp))
    return false;

  if (isa<SourceOp>(value.getDefiningOp()))
    return true;
  return llvm::all_of(value.getDefiningOp()->getOperands(),
                      [](Value v) { return isSourced(v); });
}

static llvm::SmallVector<Value> getSubjectOperands(Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For LoadOp, only the data result is considered a subject operand
    return {loadOp.getAddress()};
  }
  return llvm::to_vector(op->getOperands());
}

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

template <typename OpT>
static LogicalResult performMotion(Operation *op,
                                   std::function<OpT(Value)> buildOp) {
  for (Value result : getSubjectResults(op)) {
    OpT newOp = buildOp(result);
    inheritBB(op, newOp);
    if (newOp->getNumResults() != 1)
      return op->emitError("Expected OpT to have a single result");
    result.replaceAllUsesExcept(newOp->getResult(0), newOp);
    materializeValue(newOp->getResult(0));
  }

  for (Value operand : getSubjectOperands(op)) {
    if (isSourced(operand))
      continue;

    Operation *definingOp = operand.getDefiningOp();
    if (!isa<OpT>(definingOp))
      return op->emitError("Expected all operands to be defined by the OpT");
    if (definingOp->getNumResults() != 1)
      return op->emitError("Expected OpT to have a single result");
    definingOp->getResult(0).replaceAllUsesWith(definingOp->getOperand(0));
    definingOp->erase();
  }
  return success();
}

static bool tryMovingSup(OpBuilder &builder, PasserOp passerOp,
                         DenseSet<PasserOp> &frontiers) {
  Value passerControl = passerOp.getCtrl();
  Location passerLoc = passerOp.getLoc();

  for (Operation *targetOp : passerOp.getResult().getUsers()) {
    bool updated = false;
    TypeSwitch<Operation *>(targetOp)
        .Case<ArithOpInterface, NotOp, ForkOp, LazyForkOp, BufferOp, LoadOp>(
            [&](auto) {
              DenseSet<PasserOp> rewrittenPassers;
              bool isEligible = true;
              for (Value operand : getSubjectOperands(targetOp)) {
                if (auto passerOp =
                        dyn_cast<PasserOp>(operand.getDefiningOp())) {
                  if (passerControl && passerControl != passerOp.getCtrl()) {
                    isEligible = false;
                    break;
                  }
                  assertResultIsMaterialized(passerOp);
                  rewrittenPassers.insert(passerOp);
                  continue;
                }
                if (isSourced(operand))
                  continue;

                isEligible = false;
                break;
              }
              if (isEligible) {
                for (auto passer : rewrittenPassers)
                  frontiers.erase(passer);

                if (failed(performMotion<PasserOp>(targetOp, [&](Value v) {
                      return builder.create<PasserOp>(passerLoc, v,
                                                      passerControl);
                    }))) {
                  targetOp->emitError("Failed to perform motion for PasserOp");
                  llvm_unreachable("SpeculationV2 algorithm failed");
                }

                for (auto result : getSubjectResults(targetOp))
                  frontiers.insert(cast<PasserOp>(*result.getUsers().begin()));

                updated = true;
              }
            })
        .Default([&](Operation *op) {
          // op->dump();
        });
    if (updated) {
      return true;
    }
  }
  return false;
}

static SpecV2InterpolatorOp useIdentInterpolator(OpBuilder &builder,
                                                 Value specLoopContinue1) {
  builder.setInsertionPoint(specLoopContinue1.getDefiningOp());
  SpecV2InterpolatorOp interpolatorOp = builder.create<SpecV2InterpolatorOp>(
      specLoopContinue1.getLoc(), specLoopContinue1, specLoopContinue1);
  inheritBB(specLoopContinue1.getDefiningOp(), interpolatorOp);
  specLoopContinue1.replaceUsesWithIf(
      interpolatorOp.getResult(),
      [](OpOperand &operand) { return isa<PasserOp>(operand.getOwner()); });

  return interpolatorOp;
}

static FailureOr<SpecV2InterpolatorOp>
introduceNewInterpolator(OpBuilder &builder,
                         SpecV2InterpolatorOp interpolatorOp) {
  Value newLongOperand = nullptr;
  for (Operation *user : interpolatorOp.getLongOperand().getUsers()) {
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

  builder.setInsertionPoint(interpolatorOp);
  auto newInterpolatorOp = builder.create<SpecV2InterpolatorOp>(
      interpolatorOp.getLoc(), interpolatorOp.getShortOperand(),
      newLongOperand);
  inheritBB(interpolatorOp, newInterpolatorOp);

  return newInterpolatorOp;
}

static bool
isEligibleToSuppressorInduction(PasserOp bottomSuppressor,
                                SpecV2InterpolatorOp newInterpolator) {
  Operation *ctrlDefiningOp = bottomSuppressor.getCtrl().getDefiningOp();
  if (!isa<SpecV2InterpolatorOp>(ctrlDefiningOp))
    return false;

  auto oldInterpolator = cast<SpecV2InterpolatorOp>(ctrlDefiningOp);
  if (oldInterpolator.getShortOperand() != newInterpolator.getShortOperand())
    return false;

  Operation *upstreamOp = bottomSuppressor.getData().getDefiningOp();
  if (!isa<PasserOp>(upstreamOp))
    return false;

  auto topSuppressor = cast<PasserOp>(upstreamOp);
  Operation *topCtrlDefiningOp = topSuppressor.getCtrl().getDefiningOp();
  if (!isa<SpecV2RepeatingInitOp>(topCtrlDefiningOp))
    return false;

  auto topRepeatingInit = cast<SpecV2RepeatingInitOp>(topCtrlDefiningOp);
  if (topRepeatingInit.getOperand() != oldInterpolator.getLongOperand())
    return false;

  if (topSuppressor.getCtrl() != newInterpolator.getLongOperand())
    return false;

  return true;
}

static void runSuppressorInduction(PasserOp bottomSuppressor,
                                   SpecV2InterpolatorOp newInterpolator) {
  auto topSuppressor =
      cast<PasserOp>(bottomSuppressor.getData().getDefiningOp());
  // Perform the rewriting
  bottomSuppressor.getCtrlMutable()[0].set(newInterpolator.getResult());
  // Remove the top suppressor
  bottomSuppressor.getDataMutable()[0].set(topSuppressor.getData());
  topSuppressor->erase();
}

static void moveResolverTop(OpBuilder &builder, Value shortOperand) {
  auto oldRepeatingInitOp =
      cast<SpecV2RepeatingInitOp>(shortOperand.getDefiningOp());

  // Materialize the long operand
  materializeValue(shortOperand);

  // Move the repeating init op below the fork
  auto forkOp =
      cast<ForkOp>(*oldRepeatingInitOp.getResult().getUsers().begin());
  builder.setInsertionPoint(forkOp);
  if (performMotion<SpecV2RepeatingInitOp>(forkOp, [&](Value v) {
        return builder.create<SpecV2RepeatingInitOp>(
            oldRepeatingInitOp.getLoc(), v);
      }).failed()) {
    forkOp->emitError("Failed to perform motion for SpecV2RepeatingInitOp");
    llvm_unreachable("SpeculationV2 algorithm failed");
  }

  // Move the passer op below the fork
  auto oldPasserOp = cast<PasserOp>(forkOp.getOperand().getDefiningOp());
  builder.setInsertionPoint(forkOp);
  if (performMotion<PasserOp>(forkOp, [&](Value v) {
        return builder.create<PasserOp>(oldPasserOp.getLoc(), v,
                                        oldPasserOp.getCtrl());
      }).failed()) {
    forkOp->emitError("Failed to perform motion for PasserOp");
    llvm_unreachable("SpeculationV2 algorithm failed");
  }
}

static Value introduceSpecResolver(OpBuilder &builder,
                                   SpecV2InterpolatorOp interpolator) {
  // Confirm the context
  auto riOp = cast<SpecV2RepeatingInitOp>(
      interpolator.getShortOperand().getDefiningOp());
  auto passerOp = cast<PasserOp>(riOp.getOperand().getDefiningOp());
  // Todo: confirm the longOperand
  auto resolverOp = builder.create<SpecV2ResolverOp>(
      interpolator.getLoc(), passerOp.getData(), interpolator.getLongOperand());
  inheritBB(interpolator, resolverOp);
  interpolator.getResult().replaceAllUsesWith(resolverOp.getResult());
  interpolator->erase();
  riOp->erase();
  passerOp->erase();
  return resolverOp.getResult();
}

static Value generateSpecLoopExit(OpBuilder &builder, Value loopExit,
                                  Value confirmSpec) {
  builder.setInsertionPoint(confirmSpec.getDefiningOp());
  AndIOp andOp =
      builder.create<AndIOp>(confirmSpec.getLoc(), confirmSpec, loopExit);
  inheritBB(confirmSpec.getDefiningOp(), andOp);
  return andOp.getResult();
}

static bool isExitSuppressorUnifiable(PasserOp bottomSuppressor, Value loopExit,
                                      Value confirmSpec) {
  // Confirm the context
  Operation *ctrlDefiningOp = bottomSuppressor.getCtrl().getDefiningOp();
  ctrlDefiningOp->dump();
  if (!isa<PasserOp>(ctrlDefiningOp))
    return false;

  auto ctrlDefiningPasser = cast<PasserOp>(ctrlDefiningOp);
  if (!isEqualUnderMaterialization(ctrlDefiningPasser.getData(), loopExit))
    return false;

  if (!isEqualUnderMaterialization(ctrlDefiningPasser.getCtrl(), confirmSpec))
    return false;

  Operation *topOp = bottomSuppressor.getData().getDefiningOp();
  if (!isa<PasserOp>(topOp))
    return false;
  auto topSuppressor = cast<PasserOp>(topOp);

  return isEqualUnderMaterialization(topSuppressor.getCtrl(), confirmSpec);
}

static void unifyExitSuppressor(PasserOp bottomSuppressor, Value specLoopExit) {
  auto topSuppressor =
      cast<PasserOp>(bottomSuppressor.getData().getDefiningOp());
  bottomSuppressor.getCtrlMutable()[0].set(specLoopExit);
  bottomSuppressor.getDataMutable()[0].set(topSuppressor.getData());
  topSuppressor->erase();
}

static MergeOp replaceRIChainWithMerge(OpBuilder &builder,
                                       SpecV2RepeatingInitOp bottomRI,
                                       unsigned n) {
  SpecV2RepeatingInitOp topRI = bottomRI;
  for (unsigned i = 1; i < n; i++) {
    topRI = cast<SpecV2RepeatingInitOp>(topRI.getOperand().getDefiningOp());
  }

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

  topRI.getOperand().dump();
  BufferOp specLoopContinueTehb = builder.create<BufferOp>(
      specLoc, topRI.getOperand(), TimingInfo::break_r(), 1,
      BufferOp::ONE_SLOT_BREAK_R);
  setBB(specLoopContinueTehb, bb);
  specLoopContinueTehb->setAttr("specv2_buffer_as_sink",
                                builder.getBoolAttr(true));

  MergeOp merge = builder.create<MergeOp>(
      specLoc, llvm::ArrayRef<Value>{specLoopContinueTehb.getResult(),
                                     conditionConstant.getResult()});

  setBB(merge, bb);
  merge->setAttr("specv2_buffer_as_source", builder.getBoolAttr(true));

  // Buffer after a merge is required, which is added in the buffering pass.

  bottomRI.getResult().replaceAllUsesWith(merge.getResult());

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

  // NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  auto loopConditionOrFailure = findLoopCondition(funcOp, headBB, tailBB);
  if (failed(loopConditionOrFailure))
    return signalPassFailure();
  auto [loopCondition, isInverted] = loopConditionOrFailure.value();
  auto [loopContinue, loopExit] =
      generateLoopContinueAndExit(builder, loopCondition, isInverted);

  replaceBranches(funcOp, tailBB, isInverted, loopContinue, loopExit);

  Value specLoopContinue = loopContinue;
  Value selector = replaceLoopHeaders(funcOp, headBB, tailBB, loopContinue);

  DenseSet<PasserOp> frontiers;
  SmallVector<Value> specLoopContinues(n);
  for (unsigned i = 0; i < n; i++) {
    frontiers.clear();
    Value newSpecLoopContinue = appendRepeatingInit(builder, specLoopContinue);
    Value newSelector = appendInit(builder, newSpecLoopContinue);
    specLoopContinues[i] = newSpecLoopContinue;

    for (auto muxOp :
         llvm::make_early_inc_range(funcOp.getOps<handshake::MuxOp>())) {
      if (getLogicBB(muxOp) != headBB)
        continue;
      auto passerOpOrFailure =
          performMuxSupRewriting(muxOp, newSelector, newSpecLoopContinue);
      if (failed(passerOpOrFailure))
        return signalPassFailure();
      frontiers.insert(passerOpOrFailure.value());
    }

    if (failed(eraseOldInit(selector)))
      return signalPassFailure();

    bool frontiersUpdated;
    do {
      frontiersUpdated = false;
      for (auto passerOp : frontiers) {
        if (tryMovingSup(builder, passerOp, frontiers)) {
          frontiersUpdated = true;
          break;
        }
      }
    } while (frontiersUpdated);

    specLoopContinue = newSpecLoopContinue;
    selector = newSelector;
  }

  SpecV2InterpolatorOp interpolator =
      useIdentInterpolator(builder, specLoopContinues[0]);
  for (unsigned i = 1; i < n; i++) {
    auto newInterpolatorOrFailure =
        introduceNewInterpolator(builder, interpolator);
    if (failed(newInterpolatorOrFailure))
      return signalPassFailure();

    SpecV2InterpolatorOp newInterpolator = newInterpolatorOrFailure.value();
    for (Operation *user :
         llvm::make_early_inc_range(interpolator.getResult().getUsers())) {
      if (auto suppressor = dyn_cast<PasserOp>(user)) {
        if (isEligibleToSuppressorInduction(suppressor, newInterpolator)) {
          runSuppressorInduction(suppressor, newInterpolator);
        }
      }
    }

    if (!interpolator.getResult().use_empty()) {
      interpolator.emitError("The old interpolator still has users.");
      return signalPassFailure();
    }

    interpolator->erase();
    interpolator = newInterpolator;
  }

  moveResolverTop(builder, interpolator.getShortOperand());
  Value confirmSpec = introduceSpecResolver(builder, interpolator);

  Value specLoopExit = generateSpecLoopExit(builder, loopExit, confirmSpec);
  for (Operation *user : iterateOverPossiblyMaterializedUsers(loopExit)) {
    if (auto passer = dyn_cast<PasserOp>(user)) {
      if (passer.getCtrl() == confirmSpec) {
        for (Operation *bottomSuppressor : passer.getResult().getUsers()) {
          auto bottomPasser = cast<PasserOp>(bottomSuppressor);
          if (isExitSuppressorUnifiable(bottomPasser, loopExit, confirmSpec)) {
            unifyExitSuppressor(bottomPasser, specLoopExit);
          } else {
            bottomPasser->emitError(
                "Expected the exit suppressor to be unifiable");
          }
        }
      }
    }
  }

  // Erase unused PasserOps
  for (auto passerOp : llvm::make_early_inc_range(funcOp.getOps<PasserOp>())) {
    if (passerOp.getResult().use_empty())
      passerOp->erase();
  }

  if (variable) {
    MergeOp merge = replaceRIChainWithMerge(
        builder,
        cast<SpecV2RepeatingInitOp>(specLoopContinues[n - 1].getDefiningOp()),
        n);

    // Optimize for buffering
    auto passer = cast<PasserOp>(
        merge->getOperand(0).getDefiningOp()->getOperand(0).getDefiningOp());
    passer.getCtrlMutable()[0].set(specLoopExit);
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculationv2::createHandshakeSpeculationV2() {
  return std::make_unique<HandshakeSpeculationV2Pass>();
}
