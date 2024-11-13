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
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/Logging.h"
#include "experimental/Transforms/Speculation/PlacementFinder.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseSet.h"
#include <iostream>
#include <list>
#include <queue>
#include <string>

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculation;

namespace {
// The list to record the branches that need to be replicated
// Value: The value whose spec tag is used
// handshake::ConditionalBranchOp: The branch to replicate
// unsigned: The direction of the branch to follow
typedef std::list<std::tuple<Value, handshake::ConditionalBranchOp, unsigned>>
    CommitBranchList;

struct HandshakeSpeculationPass
    : public dynamatic::experimental::speculation::impl::
          HandshakeSpeculationBase<HandshakeSpeculationPass> {
  HandshakeSpeculationPass(const std::string &jsonPath = "",
                           bool automatic = true) {
    this->jsonPath = jsonPath;
    this->automatic = automatic;
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
  void routeCommitControl(llvm::DenseSet<Operation *> &markedPath,
                          OpOperand &currOpOperand,
                          CommitBranchList commitBranchList);

  /// Wrapper around routeCommitControl to prepare and invoke the placement
  LogicalResult prepareAndPlaceCommits();

  /// Place the SaveCommit operations and the control path
  LogicalResult prepareAndPlaceSaveCommits();

  LogicalResult placeBuffers();
};
} // namespace

template <typename T>
LogicalResult HandshakeSpeculationPass::placeUnits(Value ctrlSignal) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  for (OpOperand *operand : placements.getPlacements<T>()) {
    Operation *dstOp = operand->getOwner();
    Value srcOpResult = operand->get();

    // Create and connect the new Operation
    builder.setInsertionPoint(dstOp);
    T newOp = builder.create<T>(dstOp->getLoc(), srcOpResult, ctrlSignal);
    inheritBB(dstOp, newOp);

    // Connect the new Operation to dstOp
    // Note: srcOpResult.replaceAllUsesExcept cannot be used here
    // because the uses of srcOpResult may include a newly created
    // operand for the speculator enable signal.
    operand->set(newOp.getResult());
  }

  return success();
}

// Traverse the IR in a DFS manner. Mark all paths that lead to commit units
// by adding them to the set markedPath. Returns a true if a Commit is reached.
static bool markPathToCommitsRecursive(llvm::DenseSet<Operation *> &visited,
                                       llvm::DenseSet<Operation *> &markedPath,
                                       Operation *currOp) {
  // End traversal if currOp is already in visited set
  if (auto [_, isNewOp] = visited.insert(currOp); !isNewOp)
    return false;

  if (isa<handshake::SpecCommitOp>(currOp)) {
    // End traversal at Commits and notify that the path leads to a commit
    markedPath.insert(currOp);
    return true;
  } else {
    bool foundCommit = false;
    // Continue DFS traversal
    for (Operation *succOp : currOp->getUsers())
      foundCommit |= markPathToCommitsRecursive(visited, markedPath, succOp);

    if (foundCommit)
      markedPath.insert(currOp);

    return foundCommit;
  }
}

// Mark all paths that lead to commit units by adding them to markedPath
static void markPathToCommits(llvm::DenseSet<Operation *> &markedPath,
                              SpeculatorOp &specOp) {
  // Create visited set
  llvm::DenseSet<Operation *> visited;
  visited.insert(specOp);

  // Traverse IR starting at the speculator's output
  for (Operation *succOp : specOp.getDataOut().getUsers())
    markPathToCommitsRecursive(visited, markedPath, succOp);
}

// This recursive function traverses the IR along a marked path and creates
// a control path by replicating the branches it finds in the way. It stops
// at commits and connects them to the newly created path with value
// ctrlSignal
void HandshakeSpeculationPass::routeCommitControl(
    llvm::DenseSet<Operation *> &markedPath, OpOperand &currOpOperand,
    CommitBranchList commitBranchList) {

  // If the traversal reaches a save commit, stop it
  if (placements.containsSaveCommit(currOpOperand)) {
    return;
  }

  Operation *currOp = currOpOperand.getOwner();
  // End traversal if currOp is not in the marked path to commits
  if (!markedPath.contains(currOp))
    return;

  // Remove operation from the set to avoid visiting it twice
  markedPath.erase(currOp);

  // If the traversal reaches a speculator, stop it
  if (isa<handshake::SpeculatorOp>(currOp))
    return;

  if (auto commitOp = dyn_cast<handshake::SpecCommitOp>(currOp)) {
    // We replicate branches only if the traversal reaches a commit.
    // Because sometimes a path of branches does not reach a commit unit.
    Value ctrlSignal = specOp.getCommitCtrl();
    for (auto [valueForSpecTag, branchOp, branchDir] : commitBranchList) {
      // Replicate a branch in the control path and use new control signal.
      // To do so, a structure of two connected branches is created.
      // A speculating branch first discards the condition in case that
      // the data is not speculative. In case it is speculative, a new branch
      // is created that replicates the current branch.

      MLIRContext *ctx = &getContext();
      OpBuilder builder(ctx);
      builder.setInsertionPointAfterValue(ctrlSignal);

      // The speculating branch will discard the branch's condition token if
      // the branch output is non-speculative. Speculative tag of the token is
      // currently implicit, so the branch input itself is used at the IR
      // level.
      auto branchDiscardNonSpec =
          builder.create<handshake::SpeculatingBranchOp>(
              branchOp.getLoc(), valueForSpecTag /* specTag */,
              branchOp.getConditionOperand());
      inheritBB(specOp, branchDiscardNonSpec);

      // The replicated branch directs the control token based on the path the
      // speculative token took
      auto branchReplicated = builder.create<handshake::ConditionalBranchOp>(
          branchDiscardNonSpec->getLoc(),
          branchDiscardNonSpec.getTrueResult() /* condition */,
          ctrlSignal /* data */);
      inheritBB(specOp, branchReplicated);

      ctrlSignal = branchReplicated->getResult(branchDir);
    }
    // Connect commit to the correct control signal and end traversal
    commitOp.setOperand(1, ctrlSignal);
  } else if (auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(currOp)) {
    // Follow the two branch results with a different control signal
    for (unsigned i = 0; i <= 1; ++i) {
      for (OpOperand &dstOpOperand : branchOp->getResult(i).getUses()) {
        // Value ctrl = branchReplicated->getResult(i);
        // Copy the current list. Can be optimized by using a data structure
        // with reference
        CommitBranchList newList(commitBranchList);
        newList.push_back(
            std::tuple<Value, handshake::ConditionalBranchOp, unsigned>(
                currOpOperand.get(), branchOp, i));
        routeCommitControl(markedPath, dstOpOperand, newList);
      }
    }
  } else {
    // Continue Traversal
    for (OpResult res : currOp->getResults()) {
      for (OpOperand &dstOpOperand : res.getUses()) {
        routeCommitControl(markedPath, dstOpOperand, commitBranchList);
      }
    }
  }
}

// Check that all commits have been correctly routed
static bool areAllCommitsRouted(Backedge fakeControl) {
  Value fakeValue = static_cast<Value>(fakeControl);
  if (not fakeValue.use_empty()) {
    // fakeControl is still in use, so at least one commit is not routed
    for (Operation *user : fakeValue.getUsers()) {
      user->emitError() << "This Commit could not be routed\n";
    }
    llvm::errs() << "Error: commit routing failed.\n";
    return false;
  }
  return true;
}

LogicalResult HandshakeSpeculationPass::prepareAndPlaceCommits() {
  // Create a temporal value to connect the commits
  Value commitCtrl = specOp.getCommitCtrl();
  OpBuilder builder(&getContext());
  BackedgeBuilder edgeBuilder(builder, specOp->getLoc());
  Backedge fakeControl = edgeBuilder.get(commitCtrl.getType());

  // Place commits and connect to the fake control signal
  if (failed(placeUnits<handshake::SpecCommitOp>(fakeControl)))
    return failure();

  // Start traversal to mark the path to commits the speculator output
  llvm::DenseSet<Operation *> markedPath;
  markPathToCommits(markedPath, specOp);

  // Follow the marked path and replicate branches
  for (OpOperand &succOpOperand : specOp.getDataOut().getUses())
    routeCommitControl(markedPath, succOpOperand, {});

  // Verify that all commits are routed to a control signal
  return success(areAllCommitsRouted(fakeControl));
}

static handshake::ConditionalBranchOp findControlBranch(Operation *op) {
  handshake::FuncOp funcOp = op->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");
  auto handshakeBlocks = getLogicBBs(funcOp);
  unsigned bb = getLogicBB(op).value();

  for (auto condBrOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {
    if (auto brBB = getLogicBB(condBrOp); !brBB || brBB != bb)
      continue;

    for (Value result : condBrOp->getResults()) {
      // if (result.getType().isa<handshake::ControlType>())
      //   return condBrOp;
      for (Operation *user : result.getUsers()) {

        if (isBackedge(result, user))
          return condBrOp;
      }
    }
  }

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
  ConditionalBranchOp controlBranch = findControlBranch(specOp);
  if (controlBranch == nullptr) {
    specOp->emitError() << "Could not find backedge within speculation bb.\n";
    return failure();

    // builder.setInsertionPointAfterValue(specOp.getSCCommitCtrl());

    // SmallVector<Value, 2> mergeOperands;
    // mergeOperands.push_back(specOp.getSCSaveCtrl());
    // mergeOperands.push_back(specOp.getSCCommitCtrl());
    // auto mergeOp = builder.create<handshake::MergeOp>(
    //     specOp.getLoc(), mergeOperands);
    // inheritBB(specOp, mergeOp);

    // return placeUnits<handshake::SpecSaveCommitOp>(mergeOp.getResult());
  }

  // To connect a Save-Commit, two control signals are sent from the Speculator
  // and are merged before reaching the Save-Commit.
  // The tokens take differents paths. One (SCSaveCtrl) needs to always reach
  // the SC, the other (SCCommitCtrl) should follow the actual branches
  // similarly to the Commits
  builder.setInsertionPointAfterValue(specOp.getSCCommitCtrl());
  auto branchDiscardNonSpec = builder.create<handshake::SpeculatingBranchOp>(
      controlBranch.getLoc(), specOp.getDataOut() /* spec tag */,
      controlBranch.getConditionOperand());
  inheritBB(specOp, branchDiscardNonSpec);

  // Create a conditional branch driven by SCBranchControl from speculator
  // SCBranchControl discards the commit-like signal when speculation is correct
  auto branchDiscardCondNonMisspec =
      builder.create<handshake::ConditionalBranchOp>(
          branchDiscardNonSpec.getLoc(), specOp.getSCBranchCtrl(),
          branchDiscardNonSpec.getTrueResult());
  inheritBB(specOp, branchDiscardCondNonMisspec);

  // This branch will propagate the signal SCCommitControl according to
  // the control branch condition, which comes from branchDiscardCondNonMisSpec
  auto branchReplicated = builder.create<handshake::ConditionalBranchOp>(
      branchDiscardCondNonMisspec.getLoc(),
      branchDiscardCondNonMisspec.getTrueResult(), specOp.getSCCommitCtrl());
  inheritBB(specOp, branchReplicated);

  // We create a Merge operation to join SCCSaveCtrl and SCCommitCtrl signals
  SmallVector<Value, 2> mergeOperands;
  mergeOperands.push_back(specOp.getSCSaveCtrl());

  // Helper function to check if a value leads to a Backedge
  auto isBranchBackedge = [&](Value result) {
    return llvm::any_of(result.getUsers(), [&](Operation *user) {
      return isBackedge(result, user);
    });
  };

  // We need to send the control token to the same path that the speculative
  // token followed. Hence, if any branch output leads to a backedge, replicate
  // the branch in the SaveCommit control path.

  // Check if trueResult of controlBranch leads to a backedge (loop)
  if (isBranchBackedge(controlBranch.getTrueResult())) {
    mergeOperands.push_back(branchReplicated.getTrueResult());
  }
  // Check if falseResult of controlBranch leads to a backedge (loop)
  else if (isBranchBackedge(controlBranch.getFalseResult())) {
    mergeOperands.push_back(branchReplicated.getFalseResult());
  }
  // If neither trueResult nor falseResult leads to a backedge, handle the error
  else {
    // todo
    // mergeOperands.push_back(branchReplicated.getTrueResult());
    unsigned bb = getLogicBB(specOp).value();
    controlBranch->emitError()
        << "Could not find the backedge in the Control Branch " << bb << "\n";
    return failure();
  }

  // All the inputs to the merge operation are ready
  auto mergeOp = builder.create<handshake::MergeOp>(branchReplicated.getLoc(),
                                                    mergeOperands);
  inheritBB(specOp, mergeOp);

  // All the control logic is set up, now connect the Save-Commits with
  // the result of mergeOp
  return placeUnits<handshake::SpecSaveCommitOp>(mergeOp.getResult());
}

std::optional<Value> findControlInputToBB(Operation *op) {
  handshake::FuncOp funcOp = op->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");

  std::optional<unsigned> targetBB = getLogicBB(op);
  if (!targetBB) {
    op->emitError("Operation does not have a BB.");
    return {};
  }

  // We use the control token, which is an input to the control branch
  // as the enable signal for the speculator.
  mlir::Value ctrlSignal;
  bool isControlBranchFound = false;
  for (auto branchOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {
    // Check if the branch is in the same BB as the operation
    // specified as the location for the speculator
    if (auto brBB = getLogicBB(branchOp); !brBB || brBB != targetBB)
      continue;

    // Check if the branch targets a control token
    if (branchOp.getDataOperand().getType().isa<handshake::ControlType>()) {
      if (isControlBranchFound) {
        branchOp->emitError("Multiple control branches found in the same BB");
        return {};
      }
      ctrlSignal = branchOp.getDataOperand();
      isControlBranchFound = true;
    }
  }

  if (!isControlBranchFound) {
    funcOp->emitError("Its BB #" + std::to_string(targetBB.value()) +
                      " does not have a control branch.");
    return {};
  }

  return ctrlSignal;
}

LogicalResult HandshakeSpeculationPass::placeSpeculator() {
  MLIRContext *ctx = &getContext();

  OpOperand &operand = placements.getSpeculatorPlacement();
  Operation *dstOp = operand.getOwner();
  Value srcOpResult = operand.get();

  std::optional<Value> enableSpecIn = findControlInputToBB(dstOp);
  if (not enableSpecIn.has_value()) {
    dstOp->emitError("Control signal for speculator's enableIn not found.");
    return failure();
  }

  OpBuilder builder(ctx);
  builder.setInsertionPoint(dstOp);

  specOp = builder.create<handshake::SpeculatorOp>(dstOp->getLoc(), srcOpResult,
                                                   enableSpecIn.value());

  // Replace uses of the original source operation's result with the
  // speculator's result, except in the speculator's operands (otherwise this
  // would create a self-loop from the speculator to the speculator)
  srcOpResult.replaceAllUsesExcept(specOp.getDataOut(), specOp);

  // Assign a Basic Block to the speculator
  inheritBB(dstOp, specOp);

  // for (auto user : specOp.getDataOut().getUsers()) {
  //   if (auto forkOp = dyn_cast<handshake::ForkOp>(user)) {
  //     for (auto res : forkOp.getResults()) {
  //       for (auto &use : res.getUses()) {
  //         auto bufOp = builder.create<handshake::BufferOp>(
  //           forkOp.getLoc(), use.get(), TimingInfo::tehb(), 16);
  //         inheritBB(forkOp, bufOp);
  //         use.set(bufOp.getResult());
  //       }
  //     }
  //   } else {
  //     std::cerr << "user is not a fork" << std::endl;
  //   }
  // }

  // for (auto &use : specOp.getDataOut().getUses()) {
  //   auto bufOp = builder.create<handshake::BufferOp>(
  //     specOp.getLoc(), specOp.getDataOut(), TimingInfo::tehb(), 3);
  //   inheritBB(specOp, bufOp);
  //   use.set(bufOp.getResult());
  // }

  return success();
}

LogicalResult HandshakeSpeculationPass::placeBuffers() {
  std::cerr << "start buffer placement\n";
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  for (OpOperand *operand : placements.getPlacements<handshake::BufferOp>()) {
    Operation *dstOp = operand->getOwner();
    dstOp->dump();
    Value srcOpResult = operand->get();

    builder.setInsertionPoint(dstOp);
    handshake::BufferOp newOp = builder.create<handshake::BufferOp>(
        dstOp->getLoc(), srcOpResult, TimingInfo::tehb(), 16);
    inheritBB(dstOp, newOp);

    operand->set(newOp.getResult());
  }

  return success();
}

void HandshakeSpeculationPass::runDynamaticPass() {
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  if (failed(SpeculationPlacements::readFromJSON(
          this->jsonPath, this->placements, nameAnalysis)))
    return signalPassFailure();

  // Run automatic finding of the unit placements
  if (this->automatic) {
    PlacementFinder finder(this->placements);
    if (failed(finder.findPlacements()))
      return signalPassFailure();
  }

  if (failed(placeSpeculator()))
    return signalPassFailure();

  // Place Save operations
  // if (failed(placeUnits<handshake::SpecSaveOp>(this->specOp.getSaveCtrl())))
  //   return signalPassFailure();

  // Place Commit operations and the Commit control path
  if (failed(prepareAndPlaceCommits()))
    return signalPassFailure();

  // Place SaveCommit operations and the SaveCommit control path
  if (failed(prepareAndPlaceSaveCommits()))
    return signalPassFailure();

  if (failed(placeBuffers()))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculation::createHandshakeSpeculation(
    const std::string &jsonPath, bool automatic) {
  return std::make_unique<HandshakeSpeculationPass>(jsonPath, automatic);
}
