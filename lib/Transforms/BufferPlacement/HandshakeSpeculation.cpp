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

#include "dynamatic/Transforms/BufferPlacement/HandshakeSpeculation.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include <string>
#include <unordered_set>
#include <vector>

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

// SpeculationPlacements Methods

void SpeculationPlacements::setSpeculator(StringRef srcOpName,
                                          StringRef dstOpName) {
  this->speculator = Placement({srcOpName, dstOpName});
}

void SpeculationPlacements::addSave(StringRef srcOpName, StringRef dstOpName) {
  this->saves.insert({srcOpName.str(), dstOpName.str()});
}

void SpeculationPlacements::addCommit(StringRef srcOpName,
                                      StringRef dstOpName) {
  this->commits.insert({srcOpName.str(), dstOpName.str()});
}

void SpeculationPlacements::addSaveCommit(StringRef srcOpName,
                                          StringRef dstOpName) {
  this->saveCommits.insert({srcOpName.str(), dstOpName.str()});
}

bool SpeculationPlacements::containsCommit(StringRef srcOpName,
                                           StringRef dstOpName) {

  // Find the range of elements with the specified key
  auto range = this->commits.equal_range(srcOpName.str());

  if (this->commits.count(srcOpName.str()))
    // Iterate over the range to find the pair {srcOpName, dstOpName}
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == dstOpName.str()) {
        return true;
      }
    }
  return false;
}

void SpeculationPlacements::eraseCommit(StringRef srcOpName,
                                        StringRef dstOpName) {

  // Find the range of elements with the specified key
  auto range = this->commits.equal_range(srcOpName.str());

  // Iterate over the range and erase the pair {srcOpName, dstOpName}
  for (auto it = range.first; it != range.second; ++it) {
    if (it->second == dstOpName.str()) {
      this->commits.erase(it);
      return; // Stop iterating after the first match is erased
    }
  }
}

Placement SpeculationPlacements::getSpeculatorPlacement() {
  return this->speculator;
}

PlacementList SpeculationPlacements::getSavePlacements() {
  return PlacementList(saves.begin(), saves.end());
}

PlacementList SpeculationPlacements::getCommitPlacements() {
  return PlacementList(commits.begin(), commits.end());
}

PlacementList SpeculationPlacements::getSaveCommitPlacements() {
  return PlacementList(saveCommits.begin(), saveCommits.end());
}

// Speculation Pass Methods

HandshakeSpeculationPass::HandshakeSpeculationPass(StringRef srcOpName,
                                                   StringRef dstOpName) {
  this->srcOpName = srcOpName;
  this->dstOpName = dstOpName;
}

template <typename T>
LogicalResult
HandshakeSpeculationPass::placeUnits(const PlacementList &placements,
                                     Value ctrlSignal) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  for (const Placement &place : placements) {
    // Get source and destination operations
    Operation *srcOp = nameAnalysis.getOp(place.srcOpName);
    Operation *dstOp = nameAnalysis.getOp(place.dstOpName);

    if (!nameAnalysis.isAnalysisValid())
      return failure(); // Could not find the operations by their name

    // If the source operation has more than one result, find
    // which one is being used in dstOp
    unsigned result_idx = 0;
    Value srcOpResult = srcOp->getResult(result_idx);
    while (not llvm::is_contained(dstOp->getOperands(), srcOpResult)) {
      result_idx++;
      if (result_idx >= srcOp->getNumResults()) {
        return failure(); // srcOp and dstOp are not connected
      }
      srcOpResult = srcOp->getResult(result_idx);
    }

    // Create and connect the new Operation
    builder.setInsertionPoint(dstOp);
    T newOp = builder.create<T>(srcOp->getLoc(), srcOpResult, ctrlSignal);

    // Connect the new Operation to dstOp
    srcOpResult.replaceAllUsesExcept(newOp.getResult(), newOp);
  }

  return success();
}

// Scan all users of a result and place a Commit if the pair is in placements
void connectCommits(MLIRContext *ctx, NameAnalysis &nameAnalysis,
                    SpeculationPlacements &placements, Operation *currOp,
                    Value result, Value ctrlSignal) {
  OpBuilder builder(ctx);
  StringRef currOpName = nameAnalysis.getName(currOp);
  for (Operation *succOp : result.getUsers()) {
    StringRef succOpName = nameAnalysis.getName(succOp);
    if (placements.containsCommit(currOpName, succOpName)) {
      // Create new Commit operation
      builder.setInsertionPointAfter(currOp);
      auto commitOp = builder.create<handshake::SpecCommitOp>(
          currOp->getLoc(), result, ctrlSignal);
      // Connect the Commit operation to the destination operation
      result.replaceAllUsesExcept(commitOp.getResult(), commitOp);
      // Remove commit from the placements
      placements.eraseCommit(currOpName, succOpName);
    }
  }
}

bool HandshakeSpeculationPass::placeCommitsTraversal(
    std::unordered_set<std::string> &visited, Value ctrlSignal,
    Operation *currOp, SpeculationPlacements &placements) {

  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
  StringRef currOpName = nameAnalysis.getName(currOp);

  // If the operation is visited, end traversal
  if (visited.count(currOpName.str()))
    return false;
  visited.insert(currOpName.str());

  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // We need to keep track if we placed a commit in case that
  // a conditional branch is needed in the control path
  bool placed_commit = false;

  // Define a function to traverse successive operations and determine
  // if a commit has been placed further down or is to be placed now
  auto traverseResult = [&](Value res) {
    for (Operation *succOp : res.getUsers()) {
      StringRef succOpName = nameAnalysis.getName(succOp);
      // The traversal is cut if containsCommit returns true
      placed_commit =
          placements.containsCommit(currOpName, succOpName) or
          placeCommitsTraversal(visited, ctrlSignal, succOp, placements) or
          placed_commit;
    }
  };

  // Define a function to create a pair of connected Speculating and
  // Conditional branches. For every branch in the speculative path,
  // the commit control token path has to replicate it, so that the control
  // tokens only reach the commits that actually received speculative tokens.
  auto createControlBranches = [&](unsigned branch_result_idx) {
    builder.setInsertionPointAfterValue(ctrlSignal);

    // The Speculating Branch will discard the control token if the token
    // in the speculative path is non-speculative.
    auto branchDisc = builder.create<handshake::SpeculatingBranchOp>(
        currOp->getLoc(), currOp->getResult(0), // specTagOperand
        currOp->getOperand(0)                   // dataOperand
    ); // (conditionOperand, dataOperand)
    // TODO(asegui): add an elastic buffer before the operands

    visited.insert(nameAnalysis.getName(branchDisc).str());

    // Connect a conditional branch at the true result of branchDisc
    auto branchCond = builder.create<handshake::ConditionalBranchOp>(
        branchDisc.getLoc(),
        branchDisc->getResult(0), // Condition
        ctrlSignal                // $TrueResult
    );
    visited.insert(nameAnalysis.getName(branchCond).str());

    // Set the control signal to the result of the newly created branches
    ctrlSignal.replaceAllUsesExcept(branchCond.getResult(branch_result_idx),
                                    branchCond);
    ctrlSignal = branchCond.getResult(branch_result_idx);
  };

  if (isa<handshake::ConditionalBranchOp>(currOp)) {
    bool created_branch = false;
    // Traverse the branch's left result
    traverseResult(currOp->getResult(0));

    if (placed_commit) {
      // Set the ctrl signal at the left branch output
      createControlBranches(0);
      created_branch = true;
    }

    // Place the left branch commits
    connectCommits(ctx, nameAnalysis, placements, currOp, currOp->getResult(0),
                   ctrlSignal);

    // Right branch section
    if (created_branch) {
      // Switch the control signal to to the right branch
      ctrlSignal = ctrlSignal.getDefiningOp()->getResult(1);
    }

    // Traverse the branch's right result
    traverseResult(currOp->getResult(1));

    if (placed_commit and not created_branch) {
      // Set the ctrl signal at the right branch output
      createControlBranches(1);
    }
    // Place the right branch commits
    connectCommits(ctx, nameAnalysis, placements, currOp, currOp->getResult(1),
                   ctrlSignal);
  } else {
    // Traverse the successor in a DFS fashion
    for (OpResult res : currOp->getResults()) {
      traverseResult(res);
    }
    // Place Commit operations
    for (OpResult res : currOp->getResults()) {
      connectCommits(ctx, nameAnalysis, placements, currOp, res, ctrlSignal);
    }
  }

  return placed_commit;
}

LogicalResult
HandshakeSpeculationPass::placeCommits(SpeculationPlacements &placements,
                                       handshake::SpeculatorOp *specOp) {
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
  StringRef specOpName = nameAnalysis.getName(*specOp);

  // Create visited set
  std::unordered_set<std::string> visited;
  visited.insert(specOpName.str());

  // Start traversal at the speculator output
  for (Operation *succOp : specOp->getResult(0).getUsers()) {
    placeCommitsTraversal(
        visited,              // set with already visited operations
        specOp->getResult(2), // $CommitCtrl, commit control signal
        succOp,               // starting operation
        placements            // placements
    );
  }

  // Add an assertion to check that all commits have been placed
  assert(placements.getCommitPlacements().size() == 0);
  return success();
}

ConditionalBranchOp findControlBranch(ModuleOp modOp, Operation *BBOp) {
  handshake::FuncOp funcOp = *modOp.getOps<handshake::FuncOp>().begin();
  assert(funcOp && "funcOp not found");
  auto handshakeBlocks = getLogicBBs(funcOp);

  handshake::ConditionalBranchOp controlBranch;
  // Iterate all Ops in the BB where we want to have Speculation
  for (auto blockOp : handshakeBlocks.blocks.lookup(1)) {
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

LogicalResult
HandshakeSpeculationPass::placeSaveCommits(const PlacementList &scPlacements,
                                           SpeculatorOp *specOp) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  if (scPlacements.size() == 0)
    return success();

  // The save commits are a result of a control branch being in the BB
  // The control path for the SC needs to replicate the branch
  ConditionalBranchOp controlBranch =
      findControlBranch(getOperation(), specOp->getParentOp());

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
  return placeUnits<handshake::SpecSaveCommitOp>(scPlacements,
                                                 mergeOp.getResult());
}

LogicalResult
HandshakeSpeculationPass::placeSpeculator(const Placement &specPlace,
                                          SpeculatorOp *specOp) {
  MLIRContext *ctx = &getContext();
  // You can call `getAnalysis` from anywhere within a pass. You need it
  // to find the operation with the provided names.
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
  Operation *srcOp = nameAnalysis.getOp(specPlace.srcOpName);
  Operation *dstOp = nameAnalysis.getOp(specPlace.dstOpName);

  if (!nameAnalysis.isAnalysisValid())
    return failure();

  // Create a new operation in-between the source and the destination
  OpBuilder builder(ctx);
  // Create it before the destination (doesn't really matter where as our IR
  // doesn't have any notion of sequential execution, but it makes sense to put
  // it right before the destination)
  builder.setInsertionPoint(dstOp);
  // Here I assume that the speculator as one input and one output of the same
  // type, but it could be anything
  *specOp = builder.create<handshake::SpeculatorOp>(srcOp->getLoc(),
                                                    srcOp->getResult(0));
  // inheritBB(srcOp, specOp);

  // Replace uses of the orginal source operation's result with the speculator's
  // result, except in the speculator's operands (otherwise this would create a
  // self-loop from the speculator to the speculator)
  srcOp->getResult(0).replaceAllUsesExcept(specOp->getResult(0),
                                           (Operation *)specOp);

  return success();
}

LogicalResult HandshakeSpeculationPass::placeSpecOperations(
    SpeculationPlacements &specPlacement) {

  SpeculatorOp specOp;
  placeSpeculator(specPlacement.getSpeculatorPlacement(), &specOp);

  // Place Save operations
  placeUnits<handshake::SpecSaveOp>(specPlacement.getSavePlacements(),
                                    specOp.getResult(1) // saveCtrl
  );

  // Place Commit operations and the Commit control path
  placeCommits(specPlacement, &specOp);

  // Place SaveCommit operations and the SaveCommit control path
  placeSaveCommits(specPlacement.getSaveCommitPlacements(), &specOp);

  return success();
}

LogicalResult HandshakeSpeculationPass::findOpPlacement(
    SpeculationPlacements &specPlacement) {
  // Dummy arcs in fir.c example
  specPlacement.setSpeculator("cmpi0", "fork5");
  specPlacement.addSave("source0", "constant1");
  specPlacement.addSave("source1", "constant4");

  specPlacement.addCommit("control_merge2", "buffer6");
  specPlacement.addCommit("extsi6", "mc_load0");

  specPlacement.addSaveCommit("buffer6", "buffer7");
  specPlacement.addSaveCommit("cond_br3", "sink1");

  return success();
}

void HandshakeSpeculationPass::runDynamaticPass() {
  SpeculationPlacements specPlacement(srcOpName, dstOpName);
  findOpPlacement(specPlacement);
  placeSpecOperations(specPlacement);
  // TODO: Add --handshake-materialize-forks-sinks to compile.sh
  // Manually inherit the BB, or can I run handshake-infer-basic-blocks?
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::buffer::createHandshakeSpeculation(StringRef srcOp,
                                              StringRef dstOp) {
  return std::make_unique<HandshakeSpeculationPass>(srcOp, dstOp);
}
