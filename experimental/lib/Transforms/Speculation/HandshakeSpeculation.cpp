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
#include "experimental/Transforms/Speculation/PlacementFinder.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
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

  /// Create the control path for commit signals by replicating branches
  LogicalResult routeCommitControl();

  /// Place commit units. Use fakeControlForCommits as a temporary control
  /// signal.
  LogicalResult placeCommits();

  /// Generate the save-commit control path and return the control signal
  FailureOr<Value> generateSaveCommitCtrl();

  /// Place save-commit units.
  LogicalResult placeSaveCommits(Value ctrlSignal);

  /// Adds a spec tag to the operand/result types in the speculative region.
  /// Traverses both upstream and downstream within the region, starting from
  /// the speculator. Upstream traversal is required to cover SourceOp and
  /// ConstantOp.
  /// See the documentation for more details:
  /// docs/Speculation/AddingSpecTagsToSpecRegion.md
  LogicalResult addSpecTagToSpecRegion();

  // Add NonSpecOps to the non-speculative edges of MuxOp/CMergeOp to satisfy
  // their type requirements.
  LogicalResult addNonSpecOp();
};
} // namespace

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

/// If the value is a result of ForkOp, returns the operand of the ForkOp.
/// Otherwise, returns the value itself.
/// If the operand of the ForkOp is also a result of ForkOp, the function
/// recursively finds the top of the fork tree.
static Value findForkTreeTop(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return value;

  if (auto forkOp = dyn_cast<ForkOp>(definingOp))
    return findForkTreeTop(forkOp.getOperand());

  if (auto lazyForkOp = dyn_cast<LazyForkOp>(definingOp))
    return findForkTreeTop(lazyForkOp.getOperand());

  // TODO: We might want to handle buffer ops here to ignore buffering
  // differences, but it's not necessary for the current use case.

  return value;
}

/// Internal helper for `findUsersInForkTree`.
/// Find all users of the MLIR values in the *partial* fork tree rooted at the
/// `value`.
static void
findUsersInForkTreeTraversal(llvm::SmallVector<Operation *> &targets,
                             Value value) {
  for (Operation *user : value.getUsers()) {
    targets.push_back(user);

    if (isa<ForkOp>(user) || isa<LazyForkOp>(user)) {
      for (OpResult result : user->getResults()) {
        findUsersInForkTreeTraversal(targets, result);
      }
    }
  }
}

/// Find all users of the MLIR values in the fork tree.
static llvm::SmallVector<Operation *> findUsersInForkTree(Value value) {
  Value forkTreeTop = findForkTreeTop(value);

  llvm::SmallVector<Operation *> targets;
  // Start the traversal from the top of the fork tree
  findUsersInForkTreeTraversal(targets, forkTreeTop);
  return targets;
}

/// Returns if two values are in the same fork tree.
static bool forkTreeEquals(Value a, Value b) {
  return findForkTreeTop(a) == findForkTreeTop(b);
}

/// Finds an existing branch op that uses the given condition and data.
/// Works for both SpeculatingBranchOp and ConditionalBranchOp.
template <typename BranchOpType>
static std::optional<BranchOpType> findExistingBranch(Value condition,
                                                      Value data) {
  // Find all users of the condition (ignoring the fork differences)
  llvm::SmallVector<Operation *> users = findUsersInForkTree(condition);
  for (Operation *user : users) {
    if (auto branchOp = dyn_cast<BranchOpType>(user)) {
      // Check if the data operand is also the same (ignoring the fork
      // differences)
      if (forkTreeEquals(branchOp.getDataOperand(), data)) {
        // Found a branch that matches the condition and data
        return branchOp;
      }
    }
  }
  return std::nullopt;
}

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

  // We assume there is a direct path to each commit from either the speculator
  // or a save-commit, and so traversal ends if we reach a save-commit or a
  // speculator. See detailed documentation for full explanation of the
  // speculative region and this assumption.
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

      auto conditionOperand = branchOp.getConditionOperand();

      std::optional<SpeculatingBranchOp> branchDiscardNonSpec =
          findExistingBranch<SpeculatingBranchOp>(valueForSpecTag,
                                                  conditionOperand);
      if (!branchDiscardNonSpec.has_value()) {
        // trueResultType and falseResultType are tentative and will be updated
        // in the addSpecTag algorithm later.
        branchDiscardNonSpec = builder.create<handshake::SpeculatingBranchOp>(
            branchOp.getLoc(),
            /*trueResultType=*/conditionOperand.getType(),
            /*falseResultType=*/conditionOperand.getType(),
            /*specTag=*/valueForSpecTag, conditionOperand);
        inheritBB(specOp, *branchDiscardNonSpec);
      }

      std::optional<ConditionalBranchOp> branchReplicated =
          findExistingBranch<ConditionalBranchOp>(
              branchDiscardNonSpec->getTrueResult(), ctrlSignal);
      if (!branchReplicated.has_value()) {
        // The replicated branch directs the control token based on the path the
        // speculative token took
        branchReplicated = builder.create<handshake::ConditionalBranchOp>(
            branchDiscardNonSpec->getLoc(),
            /*condition=*/branchDiscardNonSpec->getTrueResult(),
            /*data=*/ctrlSignal);
        inheritBB(specOp, *branchReplicated);
      }

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
  // Start traversal from the speculator
  for (OpOperand &succOpOperand : specOp.getDataOut().getUses()) {
    routeCommitControlRecursive(&getContext(), specOp, arrived, succOpOperand,
                                branchTrace);
  }
  // Start traversal from save-commit units
  for (auto saveCommitOp :
       mlir::cast<FuncOp>(specOp->getParentOp()).getOps<SpecSaveCommitOp>()) {
    for (OpOperand &succOpOperand : saveCommitOp.getDataOut().getUses()) {
      branchTrace.clear();
      routeCommitControlRecursive(&getContext(), specOp, arrived, succOpOperand,
                                  branchTrace);
    }
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
  for (OpOperand *operand : placements.getPlacements<SpecCommitOp>()) {
    Operation *dstOp = operand->getOwner();
    Value srcOpResult = operand->get();

    // Create and connect the new Operation
    builder.setInsertionPoint(dstOp);
    // resultType is tentative and will be updated in the addSpecTag algorithm
    // later.
    SpecCommitOp newOp = builder.create<SpecCommitOp>(
        dstOp->getLoc(), /*resultType=*/srcOpResult.getType(),
        /*dataIn=*/srcOpResult, /*ctrl=*/fakeControlForCommits.value());
    inheritBB(dstOp, newOp);

    // Connect the new CommitOp to dstOp
    operand->set(newOp.getResult());
  }

  return success();
}

LogicalResult HandshakeSpeculationPass::placeSaveCommits(Value ctrlSignal) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Get the specified FIFO depth
  unsigned fifoDepth = placements.getSaveCommitsFifoDepth();
  if (fifoDepth == 0) {
    llvm_unreachable("Save Commit FIFO depth cannot be 0");
  }

  for (OpOperand *operand : placements.getPlacements<SpecSaveCommitOp>()) {
    Operation *dstOp = operand->getOwner();
    Value srcOpResult = operand->get();

    // Create and connect the new Operation
    builder.setInsertionPoint(dstOp);
    // resultType is tentative and will be updated in the addSpecTag algorithm
    // later.
    SpecSaveCommitOp newOp = builder.create<SpecSaveCommitOp>(
        dstOp->getLoc(), /*resultType=*/srcOpResult.getType(),
        /*dataIn=*/srcOpResult, /*ctrl=*/ctrlSignal,
        /*fifoDepth=*/fifoDepth);
    inheritBB(dstOp, newOp);

    // Connect the new SaveCommitOp to dstOp
    operand->set(newOp.getResult());
  }

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

FailureOr<Value> HandshakeSpeculationPass::generateSaveCommitCtrl() {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

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

  auto conditionOperand = controlBranch.getConditionOperand();
  // trueResultType and falseResultType are tentative and will be updated in the
  // addSpecTag algorithm later.
  auto branchDiscardCondNonSpec =
      builder.create<handshake::SpeculatingBranchOp>(
          controlBranch.getLoc(),
          /*trueResultType=*/conditionOperand.getType(),
          /*falseResultType=*/conditionOperand.getType(),
          /*specTag=*/specOp.getDataOut(), conditionOperand);
  inheritBB(specOp, branchDiscardCondNonSpec);

  // Second, discard if speculation happened but it was correct
  // Create a conditional branch driven by SCBranchControl from speculator
  // SCBranchControl discards the commit-like signal when speculation is correct
  auto branchDiscardCondNonMisspec =
      builder.create<handshake::ConditionalBranchOp>(
          branchDiscardCondNonSpec.getLoc(), specOp.getSCIsMisspec(),
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

  // The control signal is the result of the merge op.
  return mergeOp.getResult();
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

  // Get the specified FIFO depth
  unsigned fifoDepth = placements.getSpeculatorFifoDepth();
  if (fifoDepth == 0) {
    llvm_unreachable("Speculator FIFO depth cannot be 0");
  }

  // resultType is tentative and will be updated in the addSpecTag algorithm
  // later.
  specOp = builder.create<handshake::SpeculatorOp>(
      dstOp->getLoc(), /*resultType=*/srcOpResult.getType(),
      /*dataIn=*/srcOpResult, /*specIn=*/specTrigger.value(), fifoDepth);

  // Replace uses of the original source operation's result with the
  // speculator's result, except in the speculator's operands (otherwise this
  // would create a self-loop from the speculator to the speculator)
  srcOpResult.replaceAllUsesExcept(specOp.getDataOut(), specOp);

  // Assign a Basic Block to the speculator
  inheritBB(dstOp, specOp);

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

static LogicalResult
addSpecTagToSpecRegionRecursive(MLIRContext &ctx, OpOperand &opOperand,
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
    return failure();
  }

  if (visited.contains(op))
    return success();
  visited.insert(op);

  // Exceptional cases
  if (isa<handshake::SpecCommitOp>(op)) {
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

  if (auto saveCommitOp = dyn_cast<handshake::SpecSaveCommitOp>(op)) {
    if (isDownstream) {
      // Continue traversal to the dataOut
      for (auto &operand : saveCommitOp.getDataOut().getUses()) {
        if (failed(
                addSpecTagToSpecRegionRecursive(ctx, operand, true, visited)))
          return failure();
      }
    } else {
      // Continue traversal to the dataIn, skipping the controlIn
      // because control signals are not tagged
      auto &operand = saveCommitOp->getOpOperand(0);
      if (failed(addSpecTagToSpecRegionRecursive(ctx, operand, false, visited)))
        return failure();
    }

    return success();
  }

  if (auto speculatingBranchOp = dyn_cast<handshake::SpeculatingBranchOp>(op)) {
    if (isDownstream) {
      // Stop the traversal at the speculating branch
      return success();
    }

    // The upstream stream shouldn't reach the commit unit,
    // as that would indicate it originated the control signal network, which is
    // not tagged
    op->emitError("SpeculatingBranchOp should not be reached from "
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
        if (failed(
                addSpecTagToSpecRegionRecursive(ctx, operand, true, visited)))
          return failure();
      }
    } else {
      // Continue traversal to addrIn, skipping ports connected to the memory
      // controller.
      auto &operand = loadOp->getOpOperand(0);
      if (failed(addSpecTagToSpecRegionRecursive(ctx, operand, false, visited)))
        return failure();
    }

    return success();
  }

  if (isa<handshake::ControlMergeOp>(op) || isa<handshake::MuxOp>(op)) {
    if (isDownstream) {
      // Continue normal downstream traversal, including the index channel
      // (i.e., ControlMergeOp).
      for (auto result : op->getResults()) {
        for (auto &operand : result.getUses()) {
          if (failed(
                  addSpecTagToSpecRegionRecursive(ctx, operand, true, visited)))
            return failure();
        }
      }
    } else {
      // Continue upstream traversal only to the MuxOp's index channel
      if (auto muxOp = dyn_cast<handshake::MuxOp>(op)) {
        for (auto &operand : muxOp.getSelectOperand().getUses()) {
          if (failed(addSpecTagToSpecRegionRecursive(ctx, operand, false,
                                                     visited)))
            return failure();
        }
      }
    }

    return success();
  }

  // General case

  // Upstream traversal
  for (auto &operand : op->getOpOperands()) {
    // Skip the operand that is the same as the current operand
    if (isDownstream && &operand == &opOperand)
      continue;
    if (failed(addSpecTagToSpecRegionRecursive(ctx, operand, false, visited)))
      return failure();
  }

  // Downstream traversal
  for (auto result : op->getResults()) {
    for (auto &operand : result.getUses()) {
      // Skip the operand that is the same as the current operand
      if (!isDownstream && &operand == &opOperand)
        continue;
      if (failed(addSpecTagToSpecRegionRecursive(ctx, operand, true, visited)))
        return failure();
    }
  }

  return success();
}

LogicalResult HandshakeSpeculationPass::addSpecTagToSpecRegion() {
  llvm::DenseSet<Operation *> visited;
  visited.insert(specOp);

  // For the speculator, perform downstream traversal to only dataOut, skipping
  // control signals. The upstream dataIn will be handled by the recursive
  // traversal.
  for (OpOperand &opOperand : specOp.getDataOut().getUses()) {
    if (failed(addSpecTagToSpecRegionRecursive(getContext(), opOperand, true,
                                               visited)))
      return failure();
  }
  return success();
}

LogicalResult HandshakeSpeculationPass::addNonSpecOp() {
  auto funcOp = cast<FuncOp>(specOp->getParentOp());
  OpBuilder builder(&getContext());

  for (auto mergeLikeOp : funcOp.getOps<MergeLikeOpInterface>()) {
    auto dataResultType =
        mergeLikeOp.getDataResult().getType().cast<ExtraSignalsTypeInterface>();

    if (dataResultType.hasExtraSignal(EXTRA_BIT_SPEC)) {
      // This MuxOp/CMergeOp is within the speculative region.

      // Iterate over the data operands and add NonSpecOps to the
      // non-speculative edges.
      for (auto dataOperand : mergeLikeOp.getDataOperands()) {
        auto dataOperandType =
            dataOperand.getType().cast<ExtraSignalsTypeInterface>();

        if (!dataOperandType.hasExtraSignal(EXTRA_BIT_SPEC)) {
          // Create a NonSpecOp to add the spec tag to the data operand
          builder.setInsertionPointAfterValue(dataOperand);
          auto nonSpecOp = builder.create<NonSpecOp>(
              mergeLikeOp.getLoc(), dataOperand.getType(), dataOperand);
          inheritBB(mergeLikeOp, nonSpecOp);

          // Add the spec tag to the NonSpecOp's result
          if (failed(addSpecTagToValue(nonSpecOp.getResult())))
            return failure();

          // Replace the data operand with the NonSpecOp's result
          dataOperand.replaceAllUsesExcept(nonSpecOp.getResult(), nonSpecOp);
        }
      }
    }
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

  // Save operations are not supported
  if (!placements.getPlacements<SpecSaveOp>().empty()) {
    llvm::errs() << "Error: Placement of save units is not supported.\n";
    return signalPassFailure();
  }
  // Place Save operations
  // if (failed(placeUnits<handshake::SpecSaveOp>(this->specOp.getSaveCtrl())))
  //   return signalPassFailure();

  if (!placements.getPlacements<SpecSaveCommitOp>().empty()) {
    // Generate Place SaveCommit operations and the SaveCommit control path
    FailureOr<Value> saveCommitCtrl = generateSaveCommitCtrl();
    if (failed(saveCommitCtrl))
      return signalPassFailure();

    // Place SaveCommit operations
    if (failed(placeSaveCommits(saveCommitCtrl.value())))
      return signalPassFailure();
  }

  // Place Commit operations
  if (failed(placeCommits()))
    return signalPassFailure();

  // After placing all speculative units, route the commit control signals
  if (failed(routeCommitControl()))
    return signalPassFailure();

  // After placement and routing, add the spec tag to operands/results in the
  // speculative region. Skipping this update would lead to a type verification
  // error, as type-checking happens after the pass.
  if (failed(addSpecTagToSpecRegion()))
    return signalPassFailure();

  // Finally, add NonSpecOps to the non-speculative edges of MuxOp/CMergeOp
  // to satisfy their type requirements.
  if (failed(addNonSpecOp()))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculation::createHandshakeSpeculation(
    const std::string &jsonPath, bool automatic) {
  return std::make_unique<HandshakeSpeculationPass>(jsonPath, automatic);
}
