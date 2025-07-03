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
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
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

  std::optional<Value> specLoopContinue;
  std::optional<Value> specLoopExit;
  std::optional<Value> confirmSpec;

  void placeSpeculator(FuncOp &funcOp, unsigned specBB);

  void replaceBranches(FuncOp &funcOp, unsigned specBB);
  void replaceLoopHeaders(FuncOp &funcOp, unsigned specBB);
  void placeCommits(FuncOp &funcOp, unsigned specBB);
  void placeCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                             OpOperand &currOpOperand, unsigned specBB);

  void runDynamaticPass() override;
};
} // namespace

void HandshakeSpeculationV2Pass::placeSpeculator(FuncOp &funcOp,
                                                 unsigned specBB) {

  ConditionalBranchOp condBrOp = nullptr;
  for (auto condBrCandidate : funcOp.getOps<ConditionalBranchOp>()) {
    auto condBB = getLogicBB(condBrCandidate);
    if (condBB && *condBB == specBB) {
      // Found the condBr in the specBB
      condBrOp = condBrCandidate;
      break;
    }
  }
  assert(condBrOp && "Could not find any ConditionalBranchOp");

  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(condBrOp);

  Value loopContinue = condBrOp.getConditionOperand();
  Location specLoc = loopContinue.getLoc();
  ChannelType conditionType = loopContinue.getType().cast<ChannelType>();

  // Append CommitControl
  BackedgeBuilder backedgeBuilder(builder, specLoc);
  Backedge generatedConditionBackedge = backedgeBuilder.get(conditionType);

  SpecV2ResolverOp specResolverOp = builder.create<SpecV2ResolverOp>(
      specLoc, loopContinue, generatedConditionBackedge);
  inheritBB(condBrOp, specResolverOp);

  NotOp loopContinueNot = builder.create<NotOp>(specLoc, loopContinue);
  inheritBB(condBrOp, loopContinueNot);

  AndIOp andCondition = builder.create<AndIOp>(
      specLoc, loopContinueNot.getResult(), specResolverOp.getConfirmSpec());
  inheritBB(condBrOp, andCondition);

  PasserOp loopContinueSuppressor =
      builder.create<PasserOp>(specLoc, loopContinue, andCondition.getResult());
  inheritBB(condBrOp, loopContinueSuppressor);

  SourceOp conditionGenerator = builder.create<SourceOp>(specLoc);
  inheritBB(condBrOp, conditionGenerator);
  conditionGenerator->setAttr("specv2_ignore_buffer",
                              builder.getBoolAttr(true));
  ConstantOp conditionConstant = builder.create<ConstantOp>(
      specLoc, IntegerAttr::get(conditionType.getDataType(), 1),
      conditionGenerator.getResult());
  inheritBB(condBrOp, conditionConstant);
  conditionConstant->setAttr("specv2_ignore_buffer", builder.getBoolAttr(true));

  BufferOp specLoopContinueTehb = builder.create<BufferOp>(
      specLoc, loopContinueSuppressor.getResult(), TimingInfo::break_r(), 1,
      BufferOp::ONE_SLOT_BREAK_R);
  inheritBB(condBrOp, specLoopContinueTehb);
  specLoopContinueTehb->setAttr("specv2_buffer_as_sink",
                                builder.getBoolAttr(true));

  MergeOp merge = builder.create<MergeOp>(
      specLoc, llvm::ArrayRef<Value>{specLoopContinueTehb.getResult(),
                                     conditionConstant.getResult()});
  inheritBB(condBrOp, merge);
  merge->setAttr("specv2_buffer_as_source", builder.getBoolAttr(true));

  // Buffer after a merge is required, which is added in the buffering pass.

  generatedConditionBackedge.setValue(merge.getResult());

  specLoopContinue = merge.getResult();
  specLoopExit = andCondition.getResult();
  confirmSpec = specResolverOp.getConfirmSpec();
}

/// Returns operands to traverse next when placing Commit or SaveCommit.
/// For LoadOps, only data result uses are included. For StoreOp, no targets.
/// For others, all result uses.
static llvm::SmallVector<OpOperand *>
getSpecRegionTraversalTargets(Operation *op) {
  llvm::SmallVector<OpOperand *> targets;
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // Continue traversal only the data result of the LoadOp, skipping results
    // connected to the memory controller.
    for (OpOperand &dstOpOperand : loadOp.getDataResult().getUses()) {
      targets.push_back(&dstOpOperand);
    }
  } else if (isa<handshake::StoreOp>(op)) {
    // Traversal ends here; edges to the memory controller are skipped
    return {};
  } else {
    for (OpResult res : op->getResults()) {
      for (OpOperand &dstOpOperand : res.getUses()) {
        targets.push_back(&dstOpOperand);
      }
    }
  }
  return targets;
}

void HandshakeSpeculationV2Pass::replaceBranches(FuncOp &funcOp,
                                                 unsigned specBB) {
  OpBuilder builder(funcOp->getContext());

  // Replace all branches in the specBB with the speculator
  for (auto branchOp :
       llvm::make_early_inc_range(funcOp.getOps<ConditionalBranchOp>())) {
    if (getLogicBB(branchOp) != specBB)
      continue;

    // Assume trueResult is backedge
    builder.setInsertionPoint(branchOp);
    PasserOp loopSuppressor = builder.create<PasserOp>(
        branchOp.getLoc(), branchOp.getDataOperand(), specLoopContinue.value());
    inheritBB(branchOp, loopSuppressor);
    branchOp.getTrueResult().replaceAllUsesWith(loopSuppressor.getResult());

    PasserOp exitSuppressor = builder.create<PasserOp>(
        branchOp.getLoc(), branchOp.getDataOperand(), specLoopExit.value());
    inheritBB(branchOp, exitSuppressor);
    branchOp.getFalseResult().replaceAllUsesWith(exitSuppressor.getResult());

    branchOp->erase();
  }
}

void HandshakeSpeculationV2Pass::replaceLoopHeaders(FuncOp &funcOp,
                                                    unsigned specBB) {
  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(funcOp.getBodyBlock(),
                            funcOp.getBodyBlock()->begin());

  InitOp initOp = builder.create<InitOp>(specLoopContinue->getLoc(),
                                         specLoopContinue.value());
  inheritBB(specLoopContinue.value().getDefiningOp(), initOp);

  for (auto muxOp : funcOp.getOps<handshake::MuxOp>()) {
    if (getLogicBB(muxOp) != specBB)
      continue;

    assert(muxOp.getDataOperands().size() == 2 &&
           "MuxOp in specBB should have two data operands");

    Operation *definingOp = muxOp.getDataOperands()[1].getDefiningOp();
    if (!definingOp || getLogicBB(definingOp) != specBB) {
      Value entry = muxOp.getDataOperands()[1];
      muxOp.getDataOperandsMutable()[1].set(muxOp.getDataOperands()[0]);
      muxOp.getDataOperandsMutable()[0].set(entry);
    }
  }

  for (auto cmergeOp :
       llvm::make_early_inc_range(funcOp.getOps<handshake::ControlMergeOp>())) {
    if (getLogicBB(cmergeOp) != specBB)
      continue;

    assert(cmergeOp.getDataOperands().size() == 2 &&
           "ControlMergeOp in specBB should have two data operands");

    Value entry, backedge;
    Operation *definingOp = cmergeOp.getDataOperands()[1].getDefiningOp();
    if (!definingOp || getLogicBB(definingOp) != specBB) {
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
    cmergeOp.getIndex().replaceAllUsesWith(initOp.getResult());

    cmergeOp->erase();
  }
}

void HandshakeSpeculationV2Pass::placeCommitsTraversal(
    llvm::DenseSet<Operation *> &visited, OpOperand &currOpOperand,
    unsigned specBB) {
  Operation *currOp = currOpOperand.getOwner();
  OpBuilder builder(currOp->getContext());

  if (isa<handshake::PasserOp>(currOp)) {
    return;
  }

  if (isa<handshake::StoreOp>(currOp) ||
      isa<handshake::MemoryControllerOp>(currOp) ||
      isa<handshake::EndOp>(currOp)) {
    builder.setInsertionPoint(currOp);
    PasserOp commitSuppressor = builder.create<PasserOp>(
        confirmSpec.value().getLoc(), currOpOperand.get(), confirmSpec.value());
    inheritBB(currOpOperand.get().getDefiningOp(), commitSuppressor);
    currOpOperand.get().replaceAllUsesExcept(commitSuppressor.getResult(),
                                             commitSuppressor);
    return;
  }

  // currOp->dump();
  assert(getLogicBB(currOp) == specBB &&
         "Operation should be in the speculation BB");

  auto [_, isNewOp] = visited.insert(currOp);

  // End traversal if currOp is already in visited set
  if (!isNewOp)
    return;

  for (OpOperand *target : getSpecRegionTraversalTargets(currOp)) {
    placeCommitsTraversal(visited, *target, specBB);
  }
}

void HandshakeSpeculationV2Pass::placeCommits(FuncOp &funcOp, unsigned specBB) {
  llvm::DenseSet<Operation *> visited;
  Value entry = (*funcOp.getOps<handshake::InitOp>().begin()).getResult();
  for (auto &use : entry.getUses()) {
    // Start traversal from the entry point
    placeCommitsTraversal(visited, use, specBB);
  }
}

void HandshakeSpeculationV2Pass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  // NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  unsigned specBB = 2;

  placeSpeculator(funcOp, specBB);
  replaceBranches(funcOp, specBB);
  replaceLoopHeaders(funcOp, specBB);
  placeCommits(funcOp, specBB);

  // if (failed(eraseUnusedControlNetwork(funcOp, 1)))
  //   return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculationv2::createHandshakeSpeculationV2() {
  return std::make_unique<HandshakeSpeculationV2Pass>();
}
