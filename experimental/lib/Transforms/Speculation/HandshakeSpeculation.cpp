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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "experimental/Transforms/Speculation/PlacementFinder.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include <list>
#include <string>

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculation;

namespace {
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

  // In the placeCommits method, commit units are temporarily connected to
  // this value as an alternative to control signals and are subsequently
  // referenced in the routeCommitControl method.
  std::optional<Value> fakeControlForCommits;

  /// Place the operation handshake::SpeculatorOp
  LogicalResult placeSpeculator();

  /// Place the operation specified in T with the control signal ctrlSignal
  template <typename T>
  LogicalResult placeUnits(Value ctrlSignal);

  /// Create the control path for commit signals by replicating branches
  LogicalResult routeCommitControl();

  /// Place the Commit operations
  LogicalResult placeCommits();

  /// Place the SaveCommit operations and the control path
  LogicalResult prepareAndPlaceSaveCommits();
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
    // The following bug may occur in most cases:
    // (a) Consider a scenario where a control value from a buffer is passed to
    // the control branch.
    // (b) Simultaneously, a speculator uses the same control value from the
    // buffer as a trigger signal.
    // (c) A save-commit unit is positioned on the edge from the buffer to the
    // control branch.
    // (d) If we apply replaceAllUsesExcept to the value referenced by the
    // save-commit unit, the speculator will also be placed after the
    // save-commit unit, which is undesirable.
    operand->set(newOp.getResult());
  }

  return success();
}

// The list item to trace the branches that need to be replicated
struct BranchTracingItem {
  // A value whose spec tag is used to discard non spec case
  Value valueForSpecTag;
  // The branch to replicate
  handshake::ConditionalBranchOp branchOp;
  // The index of the result of the branchOp
  unsigned branchDirection;
};

// This function traverses the IR and creates a control path by replicating the
// branches it finds in the way. It stops at commits and connects them to the
// newly created path with value ctrlSignal
static void routeCommitControlRecursive(
    MLIRContext *ctx, SpeculatorOp &specOp,
    llvm::DenseSet<Operation *> &arrived, OpOperand &currOpOperand,
    const std::list<BranchTracingItem> &branchTracingList) {
  Operation *currOp = currOpOperand.getOwner();

  // End traversal if currOpOperand is already arrived
  if (arrived.contains(currOp))
    return;
  arrived.insert(currOp);

  // If the traversal reaches a speculator or a save commit unit, end traversal
  // All commit units should be found within the same loop iteration
  // And passing through these units may result in the replication of redundant
  // branches
  if (isa<handshake::SpeculatorOp>(currOp))
    return;
  if (isa<handshake::SpecSaveCommitOp>(currOp))
    return;

  if (auto commitOp = dyn_cast<handshake::SpecCommitOp>(currOp)) {
    // We replicate branches only if the traversal reaches a commit.
    // Because sometimes a path of branches does not reach a commit unit.
    Value ctrlSignal = specOp.getCommitCtrl();
    for (auto [valueForSpecTag, branchOp, branchDirection] :
         branchTracingList) {
      // Replicate a branch in the control path and use new control signal.
      // To do so, a structure of two connected branches is created.
      // A speculating branch first discards the condition in case that
      // the data is not speculative. In case it is speculative, a new branch
      // is created that replicates the current branch.

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

      // Update ctrlSignal
      ctrlSignal = branchReplicated->getResult(branchDirection);
    }
    // Connect commit to the correct control signal and end traversal
    commitOp.setOperand(1, ctrlSignal);
  } else if (auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(currOp)) {
    // Follow the two branch results with a different control signal
    for (unsigned i = 0; i <= 1; ++i) {
      for (OpOperand &dstOpOperand : branchOp->getResult(i).getUses()) {
        // Copy the current branchTracingList and add the current branch info to
        // the new list.
        // The items are referenced when the traversal hits a commit unit to
        // build the commit control network.
        // We avoid modifying the original list as it's referenced elsewhere.
        // However, copying may impact performance for larger circuits. To
        // optimize, we could use a structure that references the original list,
        // adding only the new item, like a "Cons" structure in functional
        // programming
        std::list<BranchTracingItem> newList(branchTracingList);
        newList.push_back({currOpOperand.get(), branchOp, i});

        // Continue traversal with new branchTracingList
        routeCommitControlRecursive(ctx, specOp, arrived, dstOpOperand,
                                    newList);
      }
    }
  } else {
    // Continue Traversal
    for (OpResult res : currOp->getResults()) {
      for (OpOperand &dstOpOperand : res.getUses()) {
        routeCommitControlRecursive(ctx, specOp, arrived, dstOpOperand,
                                    branchTracingList);
      }
    }
  }
}

// Check that all commits have been correctly routed
static bool areAllCommitsRouted(Value fakeControl) {
  if (not fakeControl.use_empty()) {
    // fakeControl is still in use, so at least one commit is not routed
    for (Operation *user : fakeControl.getUsers()) {
      user->emitError() << "This Commit could not be routed\n";
    }
    llvm::errs() << "Error: commit routing failed.\n";
    return false;
  }
  return true;
}

LogicalResult HandshakeSpeculationPass::routeCommitControl() {
  if (!fakeControlForCommits.has_value()) {
    llvm::errs() << "Error: fakeControlForCommits doesn't have a value. Please "
                    "place commit units first.\n";
    return failure();
  }

  llvm::DenseSet<Operation *> arrived;
  for (OpOperand &succOpOperand : specOp.getDataOut().getUses())
    routeCommitControlRecursive(&getContext(), specOp, arrived, succOpOperand,
                                {});

  // Verify that all commits are routed to a control signal
  return success(areAllCommitsRouted(fakeControlForCommits.value()));
}

LogicalResult HandshakeSpeculationPass::placeCommits() {
  // Create a temporal value to connect the commits
  Value commitCtrl = specOp.getCommitCtrl();
  OpBuilder builder(&getContext());

  // Build a temporary control value using mlir::UnrealizedConversionCastOp
  // Referenced by the routeCommitControl method later
  // Note: The BackedgeBuilder cannot be used in this context. The value
  // generated by the BackedgeBuilder must be replaced before the builder is
  // destroyed, which occurs before exiting this method.
  fakeControlForCommits =
      builder
          .create<mlir::UnrealizedConversionCastOp>(
              specOp->getLoc(), commitCtrl.getType(), ValueRange{})
          .getResult(0);

  // Place commits and connect to the fake control signal
  if (failed(
          placeUnits<handshake::SpecCommitOp>(fakeControlForCommits.value())))
    return failure();

  return success();
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
  }

  // To connect a Save-Commit, two control signals are sent from the Speculator
  // and are merged before reaching the Save-Commit.
  // The tokens take differents paths. One (SCSaveCtrl) needs to always reach
  // the SC, the other (SCCommitCtrl) should follow the actual branches
  // similarly to the Commits
  builder.setInsertionPointAfterValue(specOp.getSCCommitCtrl());

  // First, discard if speculation didn't happen
  auto branchDiscardCondNonSpec =
      builder.create<handshake::SpeculatingBranchOp>(
          controlBranch.getLoc(), specOp.getDataOut() /* spec tag */,
          controlBranch.getConditionOperand());
  inheritBB(specOp, branchDiscardCondNonSpec);

  // Second, discard if speculation happened but it was correct
  // Create a conditional branch driven by SCBranchControl from speculator
  // SCBranchControl discards the commit-like signal when speculation is correct
  auto branchDiscardCondNonMisspec =
      builder.create<handshake::ConditionalBranchOp>(
          branchDiscardCondNonSpec.getLoc(), specOp.getSCBranchCtrl(),
          branchDiscardCondNonSpec.getTrueResult());
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

std::optional<Value> findControlInputToBB(handshake::FuncOp &funcOp,
                                          unsigned targetBB) {
  // Here we fork control token to use as trigger signal to speculator.
  // The presence of a buffer between this fork and the control branch creates
  // performance issues (see detailed speculation documentation). Therefore we
  // fork control token from directly above the control branch
  mlir::Value triggerChannelOrigin;

  // Find the control branch we want to speculate on.
  // To find: Iterate over every branch, looking for 1) same bb as speculator,
  // and 2) is a control branch
  bool isControlBranchFound = false;
  for (auto branchOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {
    // Ignore branches that are not in the speculator's BB
    if (auto brBB = getLogicBB(branchOp); !brBB || brBB != targetBB)
      continue;

    // We are looking for the control branch: data should be of control type
    if (branchOp.getDataOperand().getType().isa<handshake::ControlType>()) {
      // BB should have only one control branch at most
      if (isControlBranchFound) {
        branchOp->emitError("Multiple control branches found in the BB #" +
                            std::to_string(targetBB));
        return {};
      }
      triggerChannelOrigin = branchOp.getDataOperand();
      isControlBranchFound = true;
    }
  }

  if (!isControlBranchFound) {
    funcOp->emitError("BB #" + std::to_string(targetBB) +
                      " was marked for speculation, but no corresponding "
                      "control branch was found.");
    return {};
  }

  return triggerChannelOrigin;
}

LogicalResult HandshakeSpeculationPass::placeSpeculator() {
  MLIRContext *ctx = &getContext();

  OpOperand &operand = placements.getSpeculatorPlacement();
  Operation *dstOp = operand.getOwner();
  Value srcOpResult = operand.get();

  handshake::FuncOp funcOp = dstOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");

  // Get the BB number of the operation safely
  std::optional<unsigned> targetBB = getLogicBB(dstOp);
  if (!targetBB) {
    dstOp->emitError("Operation does not have a BB.");
    return failure();
  }

  std::optional<Value> specTrigger =
      findControlInputToBB(funcOp, targetBB.value());
  if (not specTrigger.has_value()) {
    dstOp->emitError("Control signal for speculator's trigger not found.");
    return failure();
  }

  OpBuilder builder(ctx);
  builder.setInsertionPoint(dstOp);

  specOp = builder.create<handshake::SpeculatorOp>(dstOp->getLoc(), srcOpResult,
                                                   specTrigger.value());

  // Replace uses of the original source operation's result with the
  // speculator's result, except in the speculator's operands (otherwise this
  // would create a self-loop from the speculator to the speculator)
  srcOpResult.replaceAllUsesExcept(specOp.getDataOut(), specOp);

  // Assign a Basic Block to the speculator
  inheritBB(dstOp, specOp);

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
  if (failed(placeUnits<handshake::SpecSaveOp>(this->specOp.getSaveCtrl())))
    return signalPassFailure();

  // Place Commit operations
  if (failed(placeCommits()))
    return signalPassFailure();

  // Place SaveCommit operations and the SaveCommit control path
  if (failed(prepareAndPlaceSaveCommits()))
    return signalPassFailure();

  // After placing all speculative units, route the commit control signals
  if (failed(routeCommitControl()))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculation::createHandshakeSpeculation(
    const std::string &jsonPath, bool automatic) {
  return std::make_unique<HandshakeSpeculationPass>(jsonPath, automatic);
}
