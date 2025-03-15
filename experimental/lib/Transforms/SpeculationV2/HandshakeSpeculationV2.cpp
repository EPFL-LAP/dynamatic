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
#include "mlir/IR/Value.h"
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

  condBrOp->dump();
  condBrOp->erase();
}

void HandshakeSpeculationV2Pass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  Value condition = nameAnalysis.getOp("cmpi0")->getResult(0);

  placeSpeculator(funcOp, 0, 1, 2);
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculationv2::createHandshakeSpeculationV2() {
  return std::make_unique<HandshakeSpeculationV2Pass>();
}
