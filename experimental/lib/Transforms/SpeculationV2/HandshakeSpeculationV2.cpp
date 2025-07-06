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
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

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

static FailureOr<PasserOp> performMuxSupRewriting(MuxOp muxOp,
                                                  Value newSelector,
                                                  Value newSpecLoopContinue) {
  if (auto passerOp =
          dyn_cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp())) {
    muxOp.getDataOperandsMutable()[1].set(passerOp.getData());
    passerOp.getDataMutable()[0].set(muxOp.getResult());
    muxOp.getResult().replaceAllUsesExcept(passerOp.getResult(), passerOp);
    muxOp.getSelectOperandMutable()[0].set(newSelector);
    passerOp.getCtrlMutable()[0].set(newSpecLoopContinue);
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

void HandshakeSpeculationV2Pass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();
  OpBuilder builder(funcOp->getContext());

  // NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  unsigned loopHeadBB = 2, loopTailBB = 2;
  auto loopConditionOrFailure =
      findLoopCondition(funcOp, loopHeadBB, loopTailBB);
  if (failed(loopConditionOrFailure))
    return signalPassFailure();
  auto [loopCondition, isInverted] = loopConditionOrFailure.value();
  auto [loopContinue, loopExit] =
      generateLoopContinueAndExit(builder, loopCondition, isInverted);

  replaceBranches(funcOp, loopTailBB, isInverted, loopContinue, loopExit);
  Value selector =
      replaceLoopHeaders(funcOp, loopHeadBB, loopTailBB, loopContinue);

  Value specLoopContinue = loopContinue;
  Value newSpecLoopContinue = appendRepeatingInit(builder, specLoopContinue);
  Value newSelector = appendInit(builder, newSpecLoopContinue);

  DenseSet<PasserOp> frontiers;
  for (auto muxOp :
       llvm::make_early_inc_range(funcOp.getOps<handshake::MuxOp>())) {
    if (getLogicBB(muxOp) != loopHeadBB)
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
      Value passerControl = passerOp.getCtrl();
      Location passerLoc = passerOp.getLoc();

      bool updated = false;
      for (Operation *targetOp : passerOp.getResult().getUsers()) {
        TypeSwitch<Operation *>(targetOp)
            .Case<ArithOpInterface, ForkOp, LazyForkOp, BufferOp, LoadOp>(
                [&](auto) {
                  DenseSet<PasserOp> rewrittenPassers;
                  bool isEligible = true;
                  for (Value operand : getSubjectOperands(targetOp)) {
                    if (auto passerOp =
                            dyn_cast<PasserOp>(operand.getDefiningOp())) {
                      if (passerControl &&
                          passerControl != passerOp.getCtrl()) {
                        isEligible = false;
                        break;
                      }
                      rewrittenPassers.insert(passerOp);
                      continue;
                    }
                    if (isSourced(operand))
                      continue;

                    isEligible = false;
                    break;
                  }
                  if (isEligible) {
                    for (auto passer : rewrittenPassers) {
                      passer.getResult().replaceAllUsesWith(passer.getData());
                      frontiers.erase(passer);
                      passer->erase();
                    }
                    for (auto result : getSubjectResults(targetOp)) {
                      if (!result.getUses().empty()) {
                        builder.setInsertionPoint(targetOp);
                        PasserOp newPasser = builder.create<PasserOp>(
                            passerLoc, result, passerControl);
                        inheritBB(targetOp, newPasser);
                        result.replaceAllUsesExcept(newPasser.getResult(),
                                                    newPasser);
                        frontiers.insert(newPasser);
                      }
                    }
                    updated = true;
                  }
                })
            .Default([&](Operation *op) {
              // op->dump();
            });
        if (updated) {
          break;
        }
      }
      if (updated) {
        frontiersUpdated = true;
        break;
      }
    }
  } while (frontiersUpdated);
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculationv2::createHandshakeSpeculationV2() {
  return std::make_unique<HandshakeSpeculationV2Pass>();
}
