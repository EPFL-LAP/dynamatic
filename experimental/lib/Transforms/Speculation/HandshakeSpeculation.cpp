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
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "experimental/Transforms/Speculation/NewPlacementFinder.h"
#include "experimental/Transforms/Speculation/PlacementFinder.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
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

  LogicalResult placeBuffers();

  /// Update the types to include the speculative tag
  LogicalResult updateTypes();
};
} // namespace

const std::string EXTRA_BIT_SPEC = "spec";

template <>
LogicalResult HandshakeSpeculationPass::placeUnits<handshake::SpecCommitOp>(
    Value ctrlSignal) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  for (OpOperand *operand :
       placements.getPlacements<handshake::SpecCommitOp>()) {
    Operation *dstOp = operand->getOwner();
    Value srcOpResult = operand->get();

    // We need a buffer in most cases
    builder.setInsertionPoint(dstOp);
    handshake::BufferOp bufferOp = builder.create<handshake::BufferOp>(
        dstOp->getLoc(), srcOpResult, TimingInfo::tehb(), 16);
    inheritBB(dstOp, bufferOp);

    operand->set(bufferOp.getResult());
    // Create and connect the new Operation
    builder.setInsertionPoint(bufferOp);
    handshake::SpecCommitOp newOp = builder.create<handshake::SpecCommitOp>(
        bufferOp->getLoc(), bufferOp.getResult(), ctrlSignal);
    inheritBB(dstOp, newOp);

    // Connect the new Operation to dstOp
    // Note: srcOpResult.replaceAllUsesExcept cannot be used here
    // because the uses of srcOpResult may include a newly created
    // operand for the speculator enable signal.
    operand->set(newOp.getResult());
  }

  return success();
}

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

// The list item to trace the branches that need to be replicated
struct BranchTracingItem {
  // A value whose spec tag is used to discard non spec case
  Value valueForSpecTag;
  // The branch to replicate
  handshake::ConditionalBranchOp branchOp;
  // The index of the result of the branchOp
  unsigned branchDirection;

  BranchTracingItem(Value valueForSpecTag,
                    handshake::ConditionalBranchOp branchOp,
                    unsigned branchDirection)
      : valueForSpecTag(valueForSpecTag), branchOp(branchOp),
        branchDirection(branchDirection) {}
};

// This function traverses the IR and creates a control path by replicating the
// branches it finds in the way. It stops at commits and connects them to the
// newly created path with value ctrlSignal
static void
routeCommitControlRecursive(MLIRContext *ctx, SpeculatorOp &specOp,
                            llvm::DenseSet<Operation *> &arrived,
                            OpOperand &currOpOperand,
                            std::vector<BranchTracingItem> &branchTrace) {
  Operation *currOp = currOpOperand.getOwner();

  // End traversal if currOpOperand is already arrived
  if (arrived.contains(currOp))
    return;
  arrived.insert(currOp);

  // We assume there is a direct path from the speculator to all commits, and so
  // traversal ends if we reach a save-commit or a speculator. See detailed
  // documentation for full explanation of the speculative region and this
  // assumption.
  if (isa<handshake::SpeculatorOp>(currOp))
    return;
  if (isa<handshake::SpecSaveCommitOp>(currOp))
    return;

  if (auto commitOp = dyn_cast<handshake::SpecCommitOp>(currOp)) {
    // We replicate branches only if the traversal reaches a commit.
    // Because sometimes a path of branches does not reach a commit unit.
    Value ctrlSignal = specOp.getCommitCtrl();
    for (auto [valueForSpecTag, branchOp, branchDirection] : branchTrace) {
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
              branchOp.getLoc(), /*specTag=*/valueForSpecTag,
              branchOp.getConditionOperand());
      inheritBB(specOp, branchDiscardNonSpec);

      // The replicated branch directs the control token based on the path the
      // speculative token took
      auto branchReplicated = builder.create<handshake::ConditionalBranchOp>(
          branchDiscardNonSpec->getLoc(),
          /*condition=*/branchDiscardNonSpec.getTrueResult(),
          /*data=*/ctrlSignal);
      inheritBB(specOp, branchReplicated);

      // Update ctrlSignal
      ctrlSignal = branchReplicated->getResult(branchDirection);
    }
    // Connect commit to the correct control signal and end traversal
    commitOp.setOperand(1, ctrlSignal);
  } else if (auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(currOp)) {
    // Follow the two branch results with a different control signal
    for (auto [i, result] : llvm::enumerate(branchOp->getResults())) {
      // Push the current branch info to the vector
      // The items are referenced when the traversal hits a commit unit to
      // build the commit control network.
      branchTrace.emplace_back(currOpOperand.get(), branchOp, (unsigned)i);

      for (OpOperand &dstOpOperand : result.getUses()) {
        // Continue traversal with new branchTracingList
        routeCommitControlRecursive(ctx, specOp, arrived, dstOpOperand,
                                    branchTrace);
      }

      // Pop the current branch info from the vector
      // This info is no longer used
      branchTrace.pop_back();
    }
  } else {
    // Continue Traversal
    for (OpResult res : currOp->getResults()) {
      for (OpOperand &dstOpOperand : res.getUses()) {
        routeCommitControlRecursive(ctx, specOp, arrived, dstOpOperand,
                                    branchTrace);
      }
    }
  }
}

// Check that all commits have been correctly routed
static bool areAllCommitsRouted(Value fakeControl) {
  if (not fakeControl.use_empty()) {
    // fakeControl is still in use, so at least one commit is not routed
    for (Operation *user : fakeControl.getUsers())
      user->emitError() << "This Commit could not be routed\n";

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
  std::vector<BranchTracingItem> branchTrace;
  for (OpOperand &succOpOperand : specOp.getDataOut().getUses()) {
    routeCommitControlRecursive(&getContext(), specOp, arrived, succOpOperand,
                                branchTrace);
  }

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

  // First, discard if speculation didn't happen
  auto branchDiscardCondNonSpec =
      builder.create<handshake::SpeculatingBranchOp>(
          controlBranch.getLoc(), /*specTag=*/specOp.getDataOut(),
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

  handshake::BufferOp bufferOp = builder.create<handshake::BufferOp>(
      dstOp->getLoc(), enableSpecIn.value(), TimingInfo::tehb(), 16);
  inheritBB(dstOp, bufferOp);

  builder.setInsertionPoint(bufferOp);
  specOp = builder.create<handshake::SpeculatorOp>(
      bufferOp->getLoc(), srcOpResult, bufferOp.getResult());

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

  // NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
  // unsigned opIdx = 1;
  // Operation *op = nameAnalysis.getOp("control_merge1");
  // if (!op) {
  //   std::cerr << "op not found\n";
  //   return failure();
  // }
  // builder.setInsertionPoint(op);
  // Operation *prevOp = op;
  // Value prevValue = op->getOperand(opIdx);
  // for (int i = 0; i < 15; i++) {
  //   handshake::BufferOp bufOp = builder.create<handshake::BufferOp>(
  //       prevOp->getLoc(), prevValue, TimingInfo::oehb(), 4);
  //   inheritBB(op, bufOp);
  //   prevOp = bufOp;
  //   prevValue = bufOp.getResult();
  // }
  // op->setOperand(opIdx, prevValue);

  return success();
}

static LogicalResult markTypeOfValueWithSpecTag(Value value) {
  OpBuilder builder(value.getContext());

  if (auto extraSignalsType =
          value.getType().dyn_cast<handshake::ExtraSignalsTypeInterface>()) {
    if (!extraSignalsType.hasExtraSignal(EXTRA_BIT_SPEC)) {
      value.setType(extraSignalsType.addExtraSignal(
          ExtraSignal(EXTRA_BIT_SPEC, builder.getIntegerType(1))));
    }
    return success();
  }
  value.dump();
  value.getDefiningOp()->emitError("Unexpected type");
  return failure();
}

static LogicalResult updateTypesRecursive(MLIRContext &ctx,
                                          OpOperand &opOperand,
                                          bool isTraversalDown,
                                          llvm::DenseSet<Operation *> &visited,
                                          int depth) {
  Operation *op;
  if (isTraversalDown) {
    op = opOperand.getOwner();
  } else {
    op = opOperand.get().getDefiningOp();
  }
  if (!op) {
    opOperand.getOwner()->emitError("Operation does not have a BB.");
    return failure();
  }
  if (visited.contains(op))
    return success();

  visited.insert(op);

  std::cerr << "depth: " << depth << "\n";
  op->dump();
  if (isTraversalDown) {
    std::cerr << "downstream\n";
  } else {
    std::cerr << "upstream\n";
  }
  // auto operand = opOperand.get().getType();
  // operand.dump();
  // std::cerr << "pointer: " << operand.getImpl() << "\n";

  // Exceptional cases
  if (isa<handshake::SpecCommitOp>(op)) {
    if (isTraversalDown) {
      // Stop the traversal at the commit unit
      return success();
    }

    // Something went wrong because the commit unit is reached from outside
    // the speculative region
    op->emitError("SpecCommitOp should not be reached from "
                  "outside the speculative region");
    return failure();
  }

  if (auto saveCommitOp = dyn_cast<handshake::SpecSaveCommitOp>(op)) {
    if (isTraversalDown) {
      for (auto &operand : saveCommitOp.getDataOut().getUses()) {
        if (failed(markTypeOfValueWithSpecTag(operand.get())))
          return failure();
        if (failed(
                updateTypesRecursive(ctx, operand, true, visited, depth + 1)))
          return failure();
      }
    } else {
      auto &operand = saveCommitOp->getOpOperand(0);
      if (failed(markTypeOfValueWithSpecTag(operand.get())))
        return failure();
      if (failed(updateTypesRecursive(ctx, operand, false, visited, depth + 1)))
        return failure();
    }

    return success();
  }

  if (auto speculatingBranchOp = dyn_cast<handshake::SpeculatingBranchOp>(op)) {
    if (isTraversalDown) {
      // Stop the traversal at the speculating branch
      return success();
    }

    // Something went wrong because the speculating branch is reached from
    // outside the speculative region
    op->emitError("SpeculatingBranchOp should not be reached from "
                  "outside the speculative region");
    return failure();
  }

  if (auto speculatorOp = dyn_cast<handshake::SpeculatorOp>(op)) {
    // Stop the traversal at the speculator
    return success();
  }

  if (isa<handshake::StoreOp>(op)) {
    op->emitError("StoreOp should not be within the speculative region");
    return failure();
  }

  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    if (isTraversalDown) {
      for (auto &operand : loadOp->getOpResult(1).getUses()) {
        if (failed(markTypeOfValueWithSpecTag(operand.get())))
          return failure();
        if (failed(
                updateTypesRecursive(ctx, operand, true, visited, depth + 1)))
          return failure();
      }
    } else {
      auto &operand = loadOp->getOpOperand(0);
      if (failed(markTypeOfValueWithSpecTag(operand.get())))
        return failure();
      if (failed(updateTypesRecursive(ctx, operand, false, visited, depth + 1)))
        return failure();
    }

    return success();
  }

  if (isa<handshake::ControlMergeOp>(op) || isa<handshake::MuxOp>(op)) {
    // Only perform traversal to the dataResult
    MergeLikeOpInterface mergeLikeOp = llvm::cast<MergeLikeOpInterface>(op);
    for (auto &operand : mergeLikeOp.getDataResult().getUses()) {
      if (failed(markTypeOfValueWithSpecTag(operand.get())))
        return failure();
      if (failed(updateTypesRecursive(ctx, operand, true, visited, depth + 1)))
        return failure();
    }

    return success();
  }

  // Upstream traversal
  for (auto &operand : op->getOpOperands()) {
    if (isTraversalDown && &operand == &opOperand)
      continue;
    if (failed(markTypeOfValueWithSpecTag(operand.get())))
      return failure();
    if (failed(updateTypesRecursive(ctx, operand, false, visited, depth + 1)))
      return failure();
  }

  // Downstream traversal
  for (auto result : op->getResults()) {
    for (auto &operand : result.getUses()) {
      if (!isTraversalDown && &operand == &opOperand)
        continue;
      if (failed(markTypeOfValueWithSpecTag(operand.get())))
        return failure();
      if (failed(updateTypesRecursive(ctx, operand, true, visited, depth + 1)))
        return failure();
    }
  }

  return success();
}

LogicalResult HandshakeSpeculationPass::updateTypes() {
  std::cerr << "start updating types\n";
  llvm::DenseSet<Operation *> visited;
  for (OpOperand &opOperand : specOp.getDataOut().getUses()) {
    if (failed(updateTypesRecursive(getContext(), opOperand, true, visited, 0)))
      return failure();
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

  if (failed(placeBuffers()))
    return signalPassFailure();

  // After completing placement of the speculator and commit units, update the
  // types to include the speculative tag. Since type-checking occurs after this
  // pass, skipping this update would result in an error.
  if (failed(updateTypes()))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculation::createHandshakeSpeculation(
    const std::string &jsonPath, bool automatic) {
  return std::make_unique<HandshakeSpeculationPass>(jsonPath, automatic);
}
