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
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

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

static void placeSpeculator(FuncOp &funcOp, unsigned preBB, unsigned specBB,
                            unsigned postBB) {
  ControlMergeOp cmergeOp = nullptr;

  for (auto cmergeOpCandidate : funcOp.getOps<ControlMergeOp>()) {
    auto cmergeBB = getLogicBB(cmergeOpCandidate);
    if (cmergeBB && *cmergeBB == specBB) {
      // Found the cmerge in the specBB
      cmergeOp = cmergeOpCandidate;
    }
  }
  assert(cmergeOp && "Could not find the cmergeOp");

  Value trigger = nullptr;
  for (auto dataOperand : cmergeOp.getDataOperands()) {
    Operation *definingOp = dataOperand.getDefiningOp();
    if (definingOp) {
      auto operandFromBB = getLogicBB(definingOp);
      if (operandFromBB && *operandFromBB == preBB) {
        // Found the operand from the nonspecBB
        trigger = dataOperand;
        break;
      }
    }
  }
  assert(trigger && "Could not find the trigger signal");

  ConditionalBranchOp condBrOp = nullptr;
  for (auto condBrCandidate : funcOp.getOps<ConditionalBranchOp>()) {
    if (!mlir::isa<ControlType>(condBrCandidate.getDataOperand().getType()))
      continue;

    auto condBB = getLogicBB(condBrCandidate);
    if (condBB && *condBB == specBB) {
      // Found the condBr in the specBB
      condBrOp = condBrCandidate;
    }
  }
  assert(condBrOp && "Could not find the condBrOp");

  Value ctrlOut = nullptr;
  Value ctrlBackedge = nullptr;
  for (auto result : condBrOp.getResults()) {
    auto userBB = getLogicBB(*result.getUsers().begin());
    if (!userBB) {
      // Todo: the user may not belong to a logic BB
      continue;
    }
    if (*userBB == postBB) {
      ctrlOut = result;
    } else {
      ctrlBackedge = result;
    }
  }
  assert(ctrlOut && "Could not find the control output");
  assert(ctrlBackedge && "Could not find the control backedge");

  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(cmergeOp);
  SpeculatorV2Op specOp = builder.create<SpeculatorV2Op>(
      cmergeOp.getLoc(), condBrOp.getConditionOperand(), trigger);
  inheritBB(condBrOp, specOp);

  ctrlOut.replaceAllUsesWith(specOp.getCtrlOut());
  ctrlBackedge.replaceAllUsesWith(condBrOp.getDataOperand());

  condBrOp->erase();
}

static SpeculatorV2Op getSpecOp(FuncOp &funcOp) {
  return *funcOp.getOps<SpeculatorV2Op>().begin();
}

static LogicalResult placeSpecMux(FuncOp &funcOp, unsigned preBB,
                                  unsigned specBB, unsigned postBB) {
  SpeculatorV2Op specOp = getSpecOp(funcOp);

  OpBuilder builder(funcOp->getContext());
  for (auto muxOp : funcOp.getOps<MuxOp>()) {
    auto muxBB = getLogicBB(muxOp);
    if (!muxBB || *muxBB != specBB) {
      continue;
    }
    if (muxOp.getDataOperands().size() != 2) {
      muxOp.emitError("MuxOp should have 2 operands");
      return failure();
    }
    Value nonSpecIn = nullptr;
    Value specIn = nullptr;
    for (auto operand : muxOp.getDataOperands()) {
      auto operandFromBB = getLogicBB(operand.getDefiningOp());
      if (operandFromBB && *operandFromBB == preBB) {
        nonSpecIn = operand;
      } else {
        specIn = operand;
      }
    }
    if (!nonSpecIn || !specIn) {
      muxOp.emitError("Could not find the operands");
      return failure();
    }

    builder.setInsertionPoint(muxOp);
    SpecMuxV2Op specMuxOp =
        builder.create<SpecMuxV2Op>(muxOp.getLoc(), nonSpecIn.getType(),
                                    specOp.getMuxCtrl(), nonSpecIn, specIn);
    inheritBB(muxOp, specMuxOp);
    muxOp.getResult().replaceAllUsesWith(specMuxOp.getResult());
    muxOp->erase();
  }

  for (auto cmergeOp : funcOp.getOps<ControlMergeOp>()) {
    auto cmergeBB = getLogicBB(cmergeOp);
    if (!cmergeBB || *cmergeBB != specBB) {
      continue;
    }
    if (cmergeOp.getDataOperands().size() != 2) {
      cmergeOp.emitError("ControlMergeOp should have 2 operands");
      return failure();
    }
    Value nonSpecIn = nullptr;
    Value specIn = nullptr;
    for (auto operand : cmergeOp.getDataOperands()) {
      auto operandFromBB = getLogicBB(operand.getDefiningOp());
      if (operandFromBB && *operandFromBB == preBB) {
        nonSpecIn = operand;
      } else {
        specIn = operand;
      }
    }
    if (!nonSpecIn || !specIn) {
      cmergeOp.emitError("Could not find the operands");
      return failure();
    }

    builder.setInsertionPoint(cmergeOp);
    SpecMuxV2Op specMuxOp =
        builder.create<SpecMuxV2Op>(cmergeOp.getLoc(), nonSpecIn.getType(),
                                    specOp.getMuxCtrl(), nonSpecIn, specIn);
    inheritBB(cmergeOp, specMuxOp);
    cmergeOp.getDataResult().replaceAllUsesWith(specMuxOp.getResult());
    cmergeOp->erase();
  }

  return success();
}

static LogicalResult eraseDataBranches(FuncOp &funcOp, unsigned specBB,
                                       unsigned postBB) {
  SmallVector<Operation *> toErase;
  for (auto condBrOp : funcOp.getOps<ConditionalBranchOp>()) {
    if (mlir::isa<ControlType>(condBrOp.getDataOperand().getType())) {
      // Control branch is already replaced with speculator's ctrlOut
      condBrOp->emitError(
          "Unexpected case: control branch should be already replaced");
      return failure();
    }
    auto condBrBB = getLogicBB(condBrOp);
    if (!condBrBB || *condBrBB != specBB) {
      continue;
    }
    for (auto result : condBrOp.getResults()) {
      auto *user = *result.getUsers().begin();
      auto userBB = getLogicBB(user);
      if (!userBB || *userBB == postBB) {
        // todo: anti commit
        user->erase();
      } else {
        result.replaceAllUsesWith(condBrOp.getDataOperand());
      }
    }
    toErase.push_back(condBrOp);
  }

  for (auto *op : toErase) {
    op->erase();
  }
  return success();
}

static LogicalResult placeCommitUnits(FuncOp &funcOp,
                                      llvm::ArrayRef<Value> placements) {
  SpeculatorV2Op specOp = getSpecOp(funcOp);

  OpBuilder builder(funcOp->getContext());
  for (auto placement : placements) {
    builder.setInsertionPointAfterValue(placement);
    SpecCommitV2Op commitOp =
        builder.create<SpecCommitV2Op>(placement.getLoc(), placement.getType(),
                                       placement, specOp.getCommitCtrl());
    inheritBB(placement.getDefiningOp(), commitOp);
    placement.replaceAllUsesExcept(commitOp.getDataOut(), commitOp);
  }

  return success();
}

void HandshakeSpeculationV2Pass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  // Value condition = nameAnalysis.getOp("cmpi0")->getResult(0);

  placeSpeculator(funcOp, 0, 1, 2);

  if (failed(placeSpecMux(funcOp, 0, 1, 2)))
    return signalPassFailure();

  if (failed(eraseDataBranches(funcOp, 1, 2)))
    return signalPassFailure();

  StoreOp storeOp = mlir::cast<StoreOp>(nameAnalysis.getOp("store1"));
  std::vector<Value> commitPlacements = {storeOp.getAddress(),
                                         storeOp.getData()};
  if (failed(placeCommitUnits(funcOp, commitPlacements)))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculationv2::createHandshakeSpeculationV2() {
  return std::make_unique<HandshakeSpeculationV2Pass>();
}
