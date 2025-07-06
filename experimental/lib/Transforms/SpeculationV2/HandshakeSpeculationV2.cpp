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

  std::optional<Value> specLoopContinue;
  std::optional<Value> specLoopExit;
  std::optional<Value> confirmSpec;

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
                            Value loopContinue, Value loopExit) {
  OpBuilder builder(funcOp->getContext());

  // Replace all branches in the specBB with the speculator
  for (auto branchOp :
       llvm::make_early_inc_range(funcOp.getOps<ConditionalBranchOp>())) {
    if (getLogicBB(branchOp) != loopTailBB)
      continue;

    // Assume trueResult is backedge
    builder.setInsertionPoint(branchOp);
    PasserOp loopSuppressor = builder.create<PasserOp>(
        branchOp.getLoc(), branchOp.getDataOperand(), loopContinue);
    inheritBB(branchOp, loopSuppressor);
    branchOp.getTrueResult().replaceAllUsesWith(loopSuppressor.getResult());

    PasserOp exitSuppressor = builder.create<PasserOp>(
        branchOp.getLoc(), branchOp.getDataOperand(), loopExit);
    inheritBB(branchOp, exitSuppressor);
    branchOp.getFalseResult().replaceAllUsesWith(exitSuppressor.getResult());

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
  specLoopContinue.getDefiningOp()->dump();
  builder.setInsertionPoint(specLoopContinue.getDefiningOp());
  InitOp initOp =
      builder.create<InitOp>(specLoopContinue.getLoc(), specLoopContinue);
  inheritBB(specLoopContinue.getDefiningOp(), initOp);
  initOp->dump();
  return initOp.getResult();
}

static LogicalResult performMuxSupRewriting(MuxOp muxOp, Value newSelector,
                                            Value newSpecLoopContinue) {
  if (auto supOp =
          dyn_cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp())) {
    muxOp.getDataOperandsMutable()[1].set(supOp.getData());
    supOp.getDataMutable()[0].set(muxOp.getResult());
    muxOp.getResult().replaceAllUsesExcept(supOp.getResult(), supOp);
    muxOp.getSelectOperandMutable()[0].set(newSelector);
    supOp.getCtrlMutable()[0].set(newSpecLoopContinue);
    return success();
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

  replaceBranches(funcOp, loopTailBB, loopContinue, loopExit);
  Value selector =
      replaceLoopHeaders(funcOp, loopHeadBB, loopTailBB, loopContinue);

  Value specLoopContinue = loopContinue;
  Value newSpecLoopContinue = appendRepeatingInit(builder, specLoopContinue);
  Value newSelector = appendInit(builder, newSpecLoopContinue);

  for (auto muxOp :
       llvm::make_early_inc_range(funcOp.getOps<handshake::MuxOp>())) {
    if (getLogicBB(muxOp) != loopHeadBB)
      continue;
    if (failed(performMuxSupRewriting(muxOp, newSelector, newSpecLoopContinue)))
      return signalPassFailure();
  }

  if (failed(eraseOldInit(selector)))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculationv2::createHandshakeSpeculationV2() {
  return std::make_unique<HandshakeSpeculationV2Pass>();
}
