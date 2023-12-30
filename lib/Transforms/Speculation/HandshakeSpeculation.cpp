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

#include "dynamatic/Transforms/Speculation/HandshakeSpeculation.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseSet.h"
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::speculation;

HandshakeSpeculationPass::HandshakeSpeculationPass(std::string unitPositions) {
  this->unitPositions = unitPositions;
}

template <typename T>
LogicalResult HandshakeSpeculationPass::placeUnits(Value ctrlSignal) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  for (const OpPlacement p : placements.getPlacements<T>()) {
    // Create and connect the new Operation
    builder.setInsertionPoint(p.dstOp);
    T newOp = builder.create<T>(p.dstOp->getLoc(), p.srcOpResult, ctrlSignal);

    // Connect the new Operation to dstOp
    p.srcOpResult.replaceAllUsesExcept(newOp.getResult(), newOp);
  }

  return success();
}

bool HandshakeSpeculationPass::routeCommitControlTraversal(
    llvm::DenseSet<Operation *> visited, Value ctrlSignal, Operation *currOp) {
  // End traversal if already visited
  if (visited.contains(currOp))
    return false;
  visited.insert(currOp);

  bool found_commit = false;
  if (isa<handshake::SpecCommitOp>(currOp)) {
    // Connect commit to the correct control signal and end traversal
    currOp->setOperand(1, ctrlSignal);
    found_commit = true;
  } else if (isa<handshake::ConditionalBranchOp>(currOp)) {
    // Replicate a branch in the control path and use new control signal
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    builder.setInsertionPointAfterValue(ctrlSignal);

    // The Speculating Branch will discard the control token if the token
    // in the speculative path is non-speculative.
    auto branchDisc = builder.create<handshake::SpeculatingBranchOp>(
        currOp->getLoc(), currOp->getResult(0), // specTagOperand
        currOp->getOperand(0)                   // dataOperand
    ); // (conditionOperand, dataOperand)
    // TODO(asegui): add an elastic buffer before the operands
    visited.insert(branchDisc);

    // Connect a conditional branch at the true result of branchDisc
    auto branchCond = builder.create<handshake::ConditionalBranchOp>(
        branchDisc.getLoc(),
        branchDisc->getResult(0), // Condition
        ctrlSignal                // $TrueResult
    );
    visited.insert(branchCond);

    // Follow the two branch results with a different control signal
    for (int i = 0; i <= 1; ++i) {
      for (Operation *succOp : currOp->getResult(i).getUsers()) {
        Value ctrl = branchCond.getResult(i);
        bool routed = routeCommitControlTraversal(visited, ctrl, succOp);
        found_commit = found_commit || routed;
      }
    }

    if (!found_commit) {
      // Remove unused branch signal
      branchCond.erase();
      branchDisc.erase();
    }

  } else {
    // Continue Traversal
    for (Value res : currOp->getResults()) {
      for (Operation *succOp : res.getUsers()) {
        bool routed = routeCommitControlTraversal(visited, ctrlSignal, succOp);
        found_commit = found_commit || routed;
      }
    }
  }
  return found_commit;
}

LogicalResult HandshakeSpeculationPass::prepareAndPlaceCommits() {

  // Place commits and connect to the Speculator Commit Control Signal
  Value commitCtrl = specOp->getResult(2);
  if (failed(placeUnits<handshake::SpecCommitOp>(commitCtrl)))
    return failure();

  // Create visited set
  llvm::DenseSet<Operation *> visited;
  visited.insert(specOp->getResult(0).getDefiningOp());

  // Start traversal at the speculator output
  for (Operation *succOp : specOp->getResult(0).getUsers()) {
    routeCommitControlTraversal(visited, commitCtrl, succOp);
  }

  return success();
}

ConditionalBranchOp findControlBranch(ModuleOp modOp, unsigned bb) {
  handshake::FuncOp funcOp = *modOp.getOps<handshake::FuncOp>().begin();
  assert(funcOp && "funcOp not found");
  auto handshakeBlocks = getLogicBBs(funcOp);

  handshake::ConditionalBranchOp controlBranch;

  // Iterate all Ops in the BB where we want to have Speculation
  for (auto blockOp : handshakeBlocks.blocks.lookup(bb)) {
    if ((controlBranch = dyn_cast<handshake::ConditionalBranchOp>(blockOp))) {
      // ASSUMPTION: The control branch has a backedge
      for (Value result : blockOp->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (isBackedge(result, user))
            return controlBranch;
        }
      }
    }
  }
  assert(false && "Could not find control branch");
  return controlBranch;
}

LogicalResult HandshakeSpeculationPass::prepareAndPlaceSaveCommits() {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Don't do anything if there are no SaveCommits to place
  if (placements.getPlacements<handshake::SpecSaveCommitOp>().size() == 0)
    return success();

  // The save commits are a result of a control branch being in the BB
  // The control path for the SC needs to replicate the branch
  unsigned bb = getLogicBB(specOp->getParentOp()).value();
  ConditionalBranchOp controlBranch = findControlBranch(getOperation(), bb);

  // To connect a Save-Commit, two control signals are sent from the Speculator
  // and are merged before reaching the Save-Commit.
  // The tokens take differents paths. One needs to always reach the SC,
  // the other should follow the actual branches similarly to the Commits
  builder.setInsertionPointAfterValue(specOp->getResult(3));
  auto branchDisc1 = builder.create<handshake::SpeculatingBranchOp>(
      controlBranch.getLoc(), controlBranch.getResult(0), // specTagOperand
      controlBranch.getOperand(0) // Ctrl signal from the branch
  );
  // TODO(asegui): place two elastic buffers, right before the inputs

  // This branch will propagate the signal SCControl1 according to the control
  // branch condition, which comes from branchDisc1
  auto branchDisc2 = builder.create<handshake::ConditionalBranchOp>(
      branchDisc1.getLoc(), branchDisc1.getResult(0), specOp->getResult(3));
  // TODO(asegui): add an elastic buffer right after specOp->getResult(3)

  // Create a conditional branch driven by SCBranchControl from speculator
  auto branchSaveCommitCond = builder.create<handshake::ConditionalBranchOp>(
      branchDisc2.getLoc(), specOp->getResult(5), branchDisc2.getResult(0));

  // We will create a Merge operation to join SCControl1 and SCControl2
  SmallVector<Value, 2> mergeOperands;
  mergeOperands.push_back(specOp->getResult(4)); // Add SCControl2

  // We need to send the control token to the same path that the speculative
  // token followed. Hence, if any branch output leads to a backedge, replicate
  // the branch in the SaveCommit control path.
  auto isBranchBackedge = [&](Value result) {
    for (Operation *user : result.getUsers()) {
      if (isBackedge(result, user))
        return true;
    }
    return false;
  };

  // Check if trueResult of controlBranch leads to a backedge (loop)
  if (isBranchBackedge(controlBranch.getResult(0))) {
    mergeOperands.push_back(branchSaveCommitCond.getResult(0));
  }

  // Check if falseResult of controlBranch leads to a backedge (loop)
  if (isBranchBackedge(controlBranch.getResult(1))) {
    mergeOperands.push_back(branchSaveCommitCond.getResult(1));
  }

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
  // inheritBB(srcOp, specOp);

  // Replace uses of the orginal source operation's result with the speculator's
  // result, except in the speculator's operands (otherwise this would create a
  // self-loop from the speculator to the speculator)
  place.srcOpResult.replaceAllUsesExcept(specOp->getResult(0),
                                         (Operation *)specOp);

  return success();
}

// Traverse the IR and store visited Operations and Values
void findSpeculativePaths(Operation *currOp,
                          SmallPtrSet<Operation *, 4> &specOperations,
                          DenseSet<Value> &specValues) {

  if (specOperations.contains(currOp))
    return;
  specOperations.insert(currOp);

  for (OpResult res : currOp->getResults()) {
    specValues.insert(res);
    for (Operation *succOp : res.getUsers()) {
      if (not isa<handshake::ConditionalBranchOp>(succOp))
        findSpeculativePaths(succOp, specOperations, specValues);
    }
  }
}

LogicalResult HandshakeSpeculationPass::findSavePlacements(Value startValue) {
  handshake::FuncOp funcOp =
      *getOperation().getOps<handshake::FuncOp>().begin();
  assert(funcOp && "funcOp not found");
  auto handshakeBlocks = getLogicBBs(funcOp);

  llvm::SmallPtrSet<Operation *, 4> specOperations;
  llvm::DenseSet<Value> specValues;
  specValues.insert(startValue);

  for (Operation *succOp : startValue.getUsers()) {
    specOperations.insert(succOp->getParentOp());
    findSpeculativePaths(succOp, specOperations, specValues);
  }

  unsigned bb = getLogicBB(startValue.getDefiningOp()).value();
  for (Operation *blockOp : handshakeBlocks.blocks.lookup(bb)) {
    if (specOperations.contains(blockOp)) {
      // Create a save if an operation has both spec and non-spec operands
      bool has_non_spec_input = false;
      for (Value operand : blockOp->getOperands()) {
        // Check if the operand is non-speculative
        if (not specValues.count(operand)) {
          has_non_spec_input = true;
        }
      }

      if (has_non_spec_input) {
        // Create a commit for every non-speculative operand
        for (Value operand : blockOp->getOperands()) {
          if (not specValues.count(operand)) {

            // No save needed in front of Source Operations
            Operation *srcOp = operand.getDefiningOp();
            if (isa<handshake::SourceOp>(srcOp))
              continue;

            placements.addSave(operand, blockOp);
          }
        }
      }
    }
  }

  return success();
}

/*
// Traverse the IR and store visited Operations and Values
void findSpeculativePathsForCommits(Operation *currOp,
                                    SmallPtrSet<Operation *> &specOperations,
                                    DenseSet<Value> &specValues,
                                    SpeculationPlacements &placements
                                    NameAnalysis &nameAnalysis) {

  if(specOperations.contains(currOp))
    return;
  specOperations.insert(currOp);

  for(OpResult res : currOp->getResults()) {
    specValues.insert(res);
    for(Operation *succOp : res->getUsers()) {
      if(not placements.containsCommit(res, succOp)) {
        findSpeculativePathsForCommits(succOp, specOperations, specValues,
                                        placements);
      }
    }
  }
}

void getBasicBlockPredecessorArcs() {
  handshake::FuncOp funcOp = *modOp.getOps<handshake::FuncOp>().begin();
  assert(funcOp && "funcOp not found");
  auto handshakeBlocks = getLogicBBs(funcOp);
}

// Due to the out-of-order problem, additional commits are needed
void findCommitsBetweenBBs(SpeculationPlacements &placements,
                           SpeculatorOp *specOp) {

  SmallPtrSet<Operation *> specOperations;
  DenseSet<Value> specValues;
  for (Operation *succOp : specOp->getResult(0)) {
    findSpeculativePathsForCommits(succOp, specOperations, specValues,
placements, nameAnalysis);
  }


  // Get a list with all pairs {Value result, Operation user} which
  // are bridges between the starting bb and previous bb OR backedges
  // Note: a BB can be a predecessor of itself
  getBasicBlockPredecessorArcs();

  // For all bridges, if there are more than 1 different BB predecessors,
  // if at least 2 of the bridges are speculative,
  // add a commit between every bridge
  // Note: a BB can be a predecessor of itself
  // Use BB Endpoints: getBBEndpoints(Value val, BBEndpoints &endpoints)

  // Clear visited set, Find speculative paths again
  specOperations.clear();
  specValues.clear();
  for (Operation *succOp : specOp->getResult(0)) {
    findSpeculativePathsForCommits(succOp, specOperations, specValues,
placements, nameAnalysis);
  }

  // Check the bridges that have a commit. If they are not speculative,
  // remove the commit

  // Optimization: remove commit units on the back edge of specBB

  // TODO(asegui): SYNCHRONIZERS
}


// Traverse the IR to find the Commit positions within basic blocks
// Assumes that Saves have already been placed
// Haoran's code also uses this function to set a specTag. We don't do that.
void findCommitTraversal(Operation *currOp, SpeculationPlacements &placements,
                         SmallPtrSet<Operation *> &visited, NameAnalysis
&nameAnalysis) {

  if(visited.contains(currOp))
    return;
  visited.insert(currOp);

  for(OpResult res : currOp->getResults()) {
    for(Operation *succOp : res->getUsers()) {
      StringRef currOpName = nameAnalysis.getName(currOp);
      StringRef succOpName = NameAnalysis.getName(succOp);

      // Add a commit if:
      // - Memtype is MC or LSQ or FUNC_EXIT (store unit or exit unit)
      // - There is a save

      if (isa<handshake::LSQOp, handshake::MemoryControllerOp>(succOp)) {
        // TODO: verify the memory operations name
        placements.addCommit(currOpName, succOpName);
      } else if (placements.containsSave(currOpName, succOpName)) {
        // Add commit if there is a save
        placements.addCommit(currOpName, succOpName);
      } else {
        // Continue traversal
        findCommitTraversal(succOp, placements, visited, nameAnalysis);
      }
    }
  }
}

// Two kinds of commit placement algorithms are used
LogicalResult findCommitPlacements(SpeculationPlacements &placements,
                                   SpeculatorOp *specOp) {
  MLIRContext *ctx = &getContext();
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  // Find commits within basic blocks
  SmallPtrSet<Operation *> visited;
  for(Operation *succOp : specOp->getResult(0)) {
    findCommitTraversal(succOp, placements, visited, nameAnalysis);
  }

  // Find commits in between basic blocks
  findCommitsBetweenBBs(placements, specOp);
  return success();
}
*/

void HandshakeSpeculationPass::runDynamaticPass() {
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  if (failed(SpeculationPlacements::readFromJSON(
          this->unitPositions, this->placements, nameAnalysis)))
    return signalPassFailure();

  if (failed(placeSpeculator()))
    return signalPassFailure();

  // Place Save operations
  if (failed(placeUnits<handshake::SpecSaveOp>(this->specOp->getResult(1))))
    return signalPassFailure();

  // Place Commit operations and the Commit control path
  if (failed(prepareAndPlaceCommits()))
    return signalPassFailure();

  // Place SaveCommit operations and the SaveCommit control path
  if (failed(prepareAndPlaceSaveCommits()))
    return signalPassFailure();

  // TODO: Add --handshake-materialize-forks-sinks to compile.sh
  // Manually inherit the BB, or can I run handshake-infer-basic-blocks?
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::speculation::createHandshakeSpeculation(std::string unitPositions) {
  return std::make_unique<HandshakeSpeculationPass>(unitPositions);
}
