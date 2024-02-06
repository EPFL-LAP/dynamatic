//===- HandshakeSpeculation.cpp - Speculative Dataflows ---------*- C++ -*-===//
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

#include "experimental/Transforms/Speculation/HandshakeSpeculation.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/Logging.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseSet.h"
#include <string>

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculation;

namespace {
struct HandshakeSpeculationPass
    : public dynamatic::experimental::speculation::impl::
          HandshakeSpeculationBase<HandshakeSpeculationPass> {
  HandshakeSpeculationPass(const std::string &jsonPath = "") {
    this->jsonPath = jsonPath;
  }

  void runDynamaticPass() override;

private:
  SpeculationPlacements placements;
  SpeculatorOp specOp;

  /// Place the operation handshake::SpeculatorOp
  LogicalResult placeSpeculator();

  /// Place the operation specified in T with the control signal ctrlSignal
  template <typename T>
  LogicalResult placeUnits(Value ctrlSignal);

  /// Create the control path for commit signals by replicating branches
  bool routeBranchControlTraversal(llvm::DenseSet<Operation *> &visited,
                                   Value ctrlSignal, Operation *currOp);

  /// Wrapper around routeCommitControlTraversal to prepare and invoke the
  /// placement
  LogicalResult prepareAndPlaceCommits();

  /// Place the SaveCommit operations and the control path
  LogicalResult prepareAndPlaceSaveCommits();
};
} // namespace

template <typename T>
LogicalResult HandshakeSpeculationPass::placeUnits(Value ctrlSignal) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  for (const OpPlacement p : placements.getPlacements<T>()) {
    // Create and connect the new Operation
    builder.setInsertionPoint(p.dstOp);
    T newOp = builder.create<T>(p.dstOp->getLoc(), p.srcOpResult, ctrlSignal);
    inheritBB(p.dstOp, newOp);

    // Connect the new Operation to dstOp
    p.srcOpResult.replaceAllUsesExcept(newOp.getResult(), newOp);
  }

  return success();
}

// This recursive function traverses the IR and creates a control path
// by replicating the branches along the way. It stops at commits and
// connects them to the newly created control path, with value ctrlSignal
bool HandshakeSpeculationPass::routeBranchControlTraversal(
    llvm::DenseSet<Operation *> &visited, Value ctrlSignal, Operation *currOp) {
  // End traversal if already visited
  if (auto [_, isNewOp] = visited.insert(currOp); !isNewOp)
    return false;

  bool foundCommit = false;
  if (auto commitOp = dyn_cast<handshake::SpecCommitOp>(currOp)) {
    // Connect commit to the correct control signal and end traversal
    commitOp.setOperand(1, ctrlSignal);
    foundCommit = true;
  } else if (auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(currOp)) {
    // Replicate a branch in the control path and use new control signal.
    // To do so, a structure of two connected branches is created.
    // A speculating branch first discards the condition in case that
    // the data is not speculative. In case it is speculative, a new branch
    // is created that replicates the current branch.

    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointAfterValue(ctrlSignal);

    // The Speculating Branch will discard the control token if the token
    // in the speculative path is non-speculative.
    auto branchDisc = builder.create<handshake::SpeculatingBranchOp>(
        branchOp.getLoc(), branchOp.getTrueResult() /* specTag */,
        branchOp.getConditionOperand());
    visited.insert(branchDisc);

    // Connect a conditional branch at the true result of branchDisc
    auto branchCond = builder.create<handshake::ConditionalBranchOp>(
        branchDisc->getLoc(), branchDisc.getTrueResult() /* condition */,
        ctrlSignal /* data */);
    visited.insert(branchCond);

    // Follow the two branch results with a different control signal
    for (unsigned i = 0; i <= 1; ++i) {
      for (Operation *succOp : currOp->getResult(i).getUsers()) {
        Value ctrl = branchCond->getResult(i);
        bool routed = routeBranchControlTraversal(visited, ctrl, succOp);
        foundCommit |= routed;
      }
    }

    if (!foundCommit) {
      // Remove unused branch signal
      branchCond->erase();
      branchDisc->erase();
    }

  } else {
    // Continue Traversal
    for (Operation *succOp : currOp->getUsers()) {
      bool routed = routeBranchControlTraversal(visited, ctrlSignal, succOp);
      foundCommit |= routed;
    }
  }
  return foundCommit;
}

LogicalResult HandshakeSpeculationPass::prepareAndPlaceCommits() {
  // Place commits and connect to the Speculator Commit Control Signal
  Value commitCtrl = specOp.getCommitCtrl();
  if (failed(placeUnits<handshake::SpecCommitOp>(commitCtrl)))
    return failure();

  // Create visited set
  llvm::DenseSet<Operation *> visited;
  visited.insert(specOp);

  // Start traversal at the speculator output
  for (Operation *succOp : specOp.getDataOut().getUsers())
    routeBranchControlTraversal(visited, commitCtrl, succOp);

  return success();
}

static handshake::ConditionalBranchOp
findControlBranch(handshake::SpeculatorOp specOp, unsigned bb) {
  handshake::FuncOp funcOp = specOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");
  auto handshakeBlocks = getLogicBBs(funcOp);

  for (auto condBrOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {
    if (auto brBB = getLogicBB(condBrOp); !brBB || brBB != bb)
      continue;

    for (Value result : condBrOp->getResults()) {
      for (Operation *user : result.getUsers()) {

        if (isBackedge(result, user))
          return condBrOp;
      }
    }
  }
  funcOp->emitError() << "Could not find backedge within BB " << bb << "\n";
  return nullptr;
}

LogicalResult HandshakeSpeculationPass::prepareAndPlaceSaveCommits() {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Don't do anything if there are no SaveCommits to place
  if (placements.getPlacements<handshake::SpecSaveCommitOp>().empty())
    return success();

  // The save commits are a result of a control branch being in the BB
  // The control path for the SC needs to replicate the branch
  unsigned bb = getLogicBB(specOp).value();
  ConditionalBranchOp controlBranch = findControlBranch(specOp, bb);
  if (controlBranch == nullptr)
    return failure();

  // To connect a Save-Commit, two control signals are sent from the Speculator
  // and are merged before reaching the Save-Commit.
  // The tokens take differents paths. One needs to always reach the SC,
  // the other should follow the actual branches similarly to the Commits
  builder.setInsertionPointAfterValue(specOp.getSCCommitCtrl());
  auto branchDiscardNonSpec = builder.create<handshake::SpeculatingBranchOp>(
      controlBranch.getLoc(), controlBranch.getTrueResult() /* spec tag */,
      controlBranch.getConditionOperand());
  inheritBB(specOp, branchDiscardNonSpec);

  // This branch will propagate the signal SCCommitControl according to
  // the control branch condition, which comes from branchDiscardNonSpec
  auto branchDiscardCond = builder.create<handshake::ConditionalBranchOp>(
      branchDiscardNonSpec.getLoc(), branchDiscardNonSpec.getTrueResult(),
      specOp.getSCCommitCtrl());
  inheritBB(specOp, branchDiscardCond);

  // Create a conditional branch driven by SCBranchControl from speculator
  auto branchSaveCommitCond = builder.create<handshake::ConditionalBranchOp>(
      branchDiscardCond.getLoc(), specOp.getSCBranchCtrl(),
      branchDiscardCond.getTrueResult());
  inheritBB(specOp, branchSaveCommitCond);

  // We create a Merge operation to join SCCSaveCtrl and SCCommitCtrl signals
  SmallVector<Value, 2> mergeOperands;
  mergeOperands.push_back(specOp.getSCSaveCtrl());

  // We need to send the control token to the same path that the speculative
  // token followed. Hence, if any branch output leads to a backedge, replicate
  // the branch in the SaveCommit control path.
  auto isBranchBackedge = [&](Value result) {
    return llvm::any_of(result.getUsers(), [&](Operation *user) {
      return isBackedge(result, user);
    });
  };

  // Check if trueResult of controlBranch leads to a backedge (loop)
  if (isBranchBackedge(controlBranch.getTrueResult()))
    mergeOperands.push_back(branchSaveCommitCond.getTrueResult());

  // Check if falseResult of controlBranch leads to a backedge (loop)
  if (isBranchBackedge(controlBranch.getFalseResult()))
    mergeOperands.push_back(branchSaveCommitCond.getFalseResult());

  // All the inputs to the merge operation are ready
  auto mergeOp = builder.create<handshake::MergeOp>(
      branchSaveCommitCond.getLoc(), mergeOperands);

  // All the control logic is set up, now connect the Save-Commits with
  // the result of mergeOp
  return placeUnits<handshake::SpecSaveCommitOp>(mergeOp.getResult());
}

LogicalResult HandshakeSpeculationPass::placeSpeculator() {
  MLIRContext *ctx = &getContext();

  OpPlacement place = placements.getSpeculatorPlacement();

  OpBuilder builder(ctx);
  builder.setInsertionPoint(place.dstOp);

  specOp = builder.create<handshake::SpeculatorOp>(place.dstOp->getLoc(),
                                                   place.srcOpResult);

  // Replace uses of the orginal source operation's result with the speculator's
  // result, except in the speculator's operands (otherwise this would create a
  // self-loop from the speculator to the speculator)
  place.srcOpResult.replaceAllUsesExcept(specOp.getDataOut(), specOp);

  // Assign a Basic Block to the speculator
  inheritBB(place.dstOp, specOp);

  return success();
}

void HandshakeSpeculationPass::runDynamaticPass() {
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  if (failed(SpeculationPlacements::readFromJSON(
          this->jsonPath, this->placements, nameAnalysis)))
    return signalPassFailure();

  if (failed(placeSpeculator()))
    return signalPassFailure();

  // Place Save operations
  if (failed(placeUnits<handshake::SpecSaveOp>(this->specOp.getSaveCtrl())))
    return signalPassFailure();

  // Place Commit operations and the Commit control path
  if (failed(prepareAndPlaceCommits()))
    return signalPassFailure();

  // Place SaveCommit operations and the SaveCommit control path
  if (failed(prepareAndPlaceSaveCommits()))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculation::createHandshakeSpeculation(
    const std::string &jsonPath) {
  return std::make_unique<HandshakeSpeculationPass>(jsonPath);
}
