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

const std::string EXTRA_BIT_SPEC = "spec";

static LogicalResult addSpecTagToValue(Value value) {
  OpBuilder builder(value.getContext());

  // The value type must implement ExtraSignalsTypeInterface (e.g., ChannelType
  // or ControlType).
  if (auto valueType =
          value.getType().dyn_cast<handshake::ExtraSignalsTypeInterface>()) {
    // Skip if the spec tag was already added during the algorithm.
    if (!valueType.hasExtraSignal(EXTRA_BIT_SPEC)) {
      llvm::SmallVector<ExtraSignal> newExtraSignals(
          valueType.getExtraSignals());
      newExtraSignals.emplace_back(EXTRA_BIT_SPEC, builder.getIntegerType(1));
      value.setType(valueType.copyWithExtraSignals(newExtraSignals));
    }
    return success();
  }
  value.getDefiningOp()->emitError("Unexpected type");
  return failure();
}

static LogicalResult addSpecTagRecursive(OpOperand &opOperand,
                                         bool isDownstream,
                                         llvm::DenseSet<Operation *> &visited) {

  if (failed(addSpecTagToValue(opOperand.get())))
    return failure();

  Operation *op;

  // Traversal may be either upstream or downstream
  if (isDownstream) {
    // Owner is the consumer of the operand
    op = opOperand.getOwner();
  } else {
    // DefiningOp is the producer of the operand
    op = opOperand.get().getDefiningOp();
  }

  if (!op) {
    // As long as the algorithm traverses inside the speculative region,
    // all operands should have an owner and defining operation.
    llvm::errs() << "op is nullptr\n";
    llvm::errs() << "downstream: " << isDownstream << "\n";
    opOperand.get().dump();
    return failure();
  }

  op->dump();

  if (visited.contains(op))
    return success();
  visited.insert(op);

  // Exceptional cases
  if (isa<handshake::SpecCommitV2Op>(op)) {
    if (isDownstream) {
      // Stop the traversal at the commit unit
      return success();
    }

    // The upstream stream shouldn't reach the commit unit,
    // as that would indicate it originated outside the speculative region.
    op->emitError("SpecCommitOp should not be reached from "
                  "outside the speculative region");
    return failure();
  }

  if (isa<handshake::StoreOp>(op)) {
    op->emitError("StoreOp should not be within the speculative region");
    return failure();
  }

  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    if (isDownstream) {
      // Continue traversal to dataOut, skipping ports connected to the memory
      // controller.
      for (auto &operand : loadOp->getOpResult(1).getUses()) {
        if (failed(addSpecTagRecursive(operand, true, visited)))
          return failure();
      }
    } else {
      // Continue traversal to addrIn, skipping ports connected to the memory
      // controller.
      auto &operand = loadOp->getOpOperand(0);
      if (failed(addSpecTagRecursive(operand, false, visited)))
        return failure();
    }

    return success();
  }

  if (auto specMuxOp = dyn_cast<handshake::SpecMuxV2Op>(op)) {
    if (isDownstream) {
      for (auto &operand : specMuxOp.getDataOut().getUses()) {
        if (failed(addSpecTagRecursive(operand, true, visited)))
          return failure();
      }
    } else {
      if (failed(addSpecTagRecursive(*specMuxOp.getSpecIn().getUses().begin(),
                                     false, visited)))
        return failure();
    }

    return success();
  }

  // General case

  // Upstream traversal
  for (auto &operand : op->getOpOperands()) {
    // Skip the operand that is the same as the current operand
    if (isDownstream && &operand == &opOperand)
      continue;
    if (failed(addSpecTagRecursive(operand, false, visited)))
      return failure();
  }

  // Downstream traversal
  for (auto result : op->getResults()) {
    if (result.getUses().empty()) {
      if (failed(addSpecTagToValue(result)))
        return failure();
      continue;
    }
    for (auto &operand : result.getUses()) {
      // Skip the operand that is the same as the current operand
      if (!isDownstream && &operand == &opOperand)
        continue;
      if (failed(addSpecTagRecursive(operand, true, visited)))
        return failure();
    }
  }

  return success();
}

static LogicalResult addSpecTag(FuncOp &funcOp) {
  SpeculatorV2Op specOp = getSpecOp(funcOp);

  llvm::DenseSet<Operation *> visited;
  visited.insert(specOp);

  if (failed(addSpecTagRecursive(*specOp.getCondition().getUses().begin(),
                                 false, visited)))
    return failure();
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

  if (failed(addSpecTag(funcOp)))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculationv2::createHandshakeSpeculationV2() {
  return std::make_unique<HandshakeSpeculationV2Pass>();
}
