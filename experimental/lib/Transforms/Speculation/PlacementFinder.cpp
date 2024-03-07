//===- PlacementFinder.cpp - Automatic speculation units finder -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class and methods for automatic finding of
// speculative units positions.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Speculation/PlacementFinder.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Logging.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculation;

PlacementFinder::PlacementFinder(SpeculationPlacements &placements)
    : placements(placements) {
  OpOperand &specPos = placements.getSpeculatorPlacement();
  assert(specPos.getOwner() && "Speculator position is undefined");
}

void PlacementFinder::clearPlacements() {
  OpOperand &specPos = placements.getSpeculatorPlacement();
  this->placements = SpeculationPlacements(specPos);
}

//===----------------------------------------------------------------------===//
// Save Units Finder Methods
//===----------------------------------------------------------------------===//

// Recursively traverse the IR until reaching branches and store visited values
// The values are stored in the set specValues that is passed by reference
static void markSpeculativePathsForSaves(Operation *currOp,
                                         DenseSet<Value> &specValues) {
  // End traversal when reaching a branch, because save units are only
  // placed inside the speculation BB
  if (isa<handshake::ConditionalBranchOp>(currOp))
    return;

  for (OpResult res : currOp->getResults()) {
    if (specValues.contains(res))
      continue;
    specValues.insert(res);
    for (Operation *succOp : res.getUsers()) {
      markSpeculativePathsForSaves(succOp, specValues);
    }
  }
}

// Save units are needed where speculative tokens can interact with
// non-speculative tokens. Updates `placements` with the Save placements
LogicalResult PlacementFinder::findSavePositions() {
  OpOperand &specPos = placements.getSpeculatorPlacement();
  handshake::FuncOp funcOp =
      specPos.getOwner()->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");
  auto handshakeBlocks = getLogicBBs(funcOp);

  // Mark all values that are speculative in the speculation BB
  llvm::DenseSet<Value> specValues;
  specValues.insert(specPos.get());
  markSpeculativePathsForSaves(specPos.getOwner(), specValues);

  // Iterate all operations in the speculation BB
  std::optional<unsigned> specBB = getLogicBB(specPos.getOwner());
  if (!specBB) {
    specPos.getOwner()->emitError("Operation does not have a BB.");
    return failure();
  }

  for (Operation *blockOp : handshakeBlocks.blocks[specBB.value()]) {
    // Create a save if an operation has both spec and non-spec operands
    bool hasNonSpecInput = false;
    bool hasSpecInput = false;
    for (Value operand : blockOp->getOperands()) {
      if (specValues.contains(operand))
        hasSpecInput = true;
      else
        hasNonSpecInput = true;
    }

    if (hasSpecInput && hasNonSpecInput) {
      for (OpOperand &operand : blockOp->getOpOperands()) {
        // Create a Save for every non-speculative operand
        if (!specValues.contains(operand.get())) {
          // No save needed in front of Source Operations
          if (isa<handshake::SourceOp>(operand.get().getDefiningOp()))
            continue;

          placements.addSave(operand);
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Commit Units Finder Methods
//===----------------------------------------------------------------------===//

// Recursively traverse the IR in a DFS way to find the placements of Commit
// units. A commit unit before (1) an exit unit; (2) a store unit; (3) a save
// unit if speculative tokens can reach them. Updates the `placements`
void PlacementFinder::findCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                                           Operation *currOp) {
  // End traversal if currOp is already in visited set
  if (auto [_, isNewOp] = visited.insert(currOp); !isNewOp)
    return;

  for (Operation *succOp : currOp->getUsers()) {
    for (OpOperand &dstOpOperand : succOp->getOpOperands()) {
      // Iterate the operands of succOp that are results of currOp
      if (dstOpOperand.get().getDefiningOp() == currOp) {
        if (placements.containsSave(dstOpOperand)) {
          // A Commit is needed in front of Save Operations. To allow for
          // multiple loop speculation, SaveCommit units are used instead of
          // consecutive Commit-Save units.
          placements.addSaveCommit(dstOpOperand);
          placements.eraseSave(dstOpOperand);
        } else if (isa<handshake::MemoryOpInterface, handshake::LoadOpInterface,
                       handshake::StoreOpInterface>(succOp)) {
          // A commit is needed in front of memory operations
          placements.addCommit(dstOpOperand);
        } else if (isa<handshake::EndOp>(succOp)) {
          // A commit is needed in front of the end/exit operation
          placements.addCommit(dstOpOperand);
        } else {
          findCommitsTraversal(visited, succOp);
        }
      }
    }
  }
}

namespace {
// Define a Control-Flow Graph Edge as the pair (srcOpResult, *dstOp)
using CFGEdge = OpOperand;

// Define a comparator between BBEndpoints
struct EndpointComparator {
  bool operator()(const BBEndpoints &a, const BBEndpoints &b) const {
    if (a.srcBB != b.srcBB)
      return a.srcBB < b.srcBB;
    return a.dstBB < b.dstBB;
  }
};

// Define a map from BBEndpoints to the CFGEdges that connect the BBs
using BBEndpointsMap =
    std::map<BBEndpoints, std::vector<CFGEdge *>, EndpointComparator>;

// Data structure to hold all arcs leading to a single BB predecessor
// Note: srcBB and dstBB can be equal when the arcs are Backedges
struct BBArc {
  unsigned srcBB;
  unsigned dstBB;
  std::vector<CFGEdge *> edges;
};

// Define a map from a BB's number to the BBArcs that lead to predecessor BBs
using BBtoPredecessorArcsMap = llvm::DenseMap<unsigned, std::vector<BBArc>>;
} // namespace

// Calculate the BBArcs that lead to predecessor BBs within funcOp
// Returns a map from each BB number to a vector of BBArcs
static BBtoPredecessorArcsMap getPredecessorArcs(handshake::FuncOp funcOp) {
  BBEndpointsMap endpointEdges;
  // Traverse all operations within funcOp to find edges between BBs, including
  // self-edges, and save them in a map from the Endpoints to the edges
  funcOp->walk([&](Operation *op) {
    for (CFGEdge &edge : op->getOpOperands()) {
      BBEndpoints endpoints;
      // Store the edge if it is a Backedge or connects two different BBs
      if (isBackedge(edge.get(), op, &endpoints) or
          endpoints.srcBB != endpoints.dstBB) {
        endpointEdges[endpoints].push_back(&edge);
      }
    }
  });

  // Join all predecessors of a BB
  BBtoPredecessorArcsMap predecessorArcs;
  for (const auto &[endpoints, edges] : endpointEdges) {
    BBArc arc;
    arc.srcBB = endpoints.srcBB;
    arc.dstBB = endpoints.dstBB;
    arc.edges = edges;
    predecessorArcs[endpoints.dstBB].push_back(arc);
  }

  return predecessorArcs;
}

// DFS traversal to mark all operations that lead to Commit units
// The set markedPaths is passed by reference and is updated with
// the OpPlacements (pair value-operation) that are traversed
static void
markSpeculativePathsForCommits(Operation *currOp,
                               SpeculationPlacements &placements,
                               llvm::DenseSet<CFGEdge *> &markedEdges) {
  for (Operation *succOp : currOp->getUsers()) {
    for (CFGEdge &edge : succOp->getOpOperands()) {
      // Iterate only the operands of succOp that are results from currOp
      if (edge.get().getDefiningOp() == currOp) {
        if (!markedEdges.count(&edge)) {
          markedEdges.insert(&edge);
          // Stop traversal if a commit is reached
          if (!placements.containsCommit(edge))
            markSpeculativePathsForCommits(succOp, placements, markedEdges);
        }
      }
    }
  }
}

// Find the placements of Commit units in between BBs, that are needed to
// avoid two control-only tokens going out of order. Updates the `placements`
void PlacementFinder::findCommitsBetweenBBs() {
  OpOperand &specPos = placements.getSpeculatorPlacement();
  auto funcOp = specPos.getOwner()->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");

  // Whenever a BB has two speculative inputs, commit units are needed to
  // avoid tokens going out-of-order. First, the block predecessor arcs are
  // found
  BBtoPredecessorArcsMap bbToPredecessorArcs = getPredecessorArcs(funcOp);

  // Mark the speculative edges. The set speculativeEdges is passed by
  // reference
  llvm::DenseSet<CFGEdge *> speculativeEdges;
  markSpeculativePathsForCommits(specPos.getOwner(), placements,
                                 speculativeEdges);

  // Iterate all BBs to check if commits are needed
  for (const auto &[bb, predecessorArcs] : bbToPredecessorArcs) {
    // Count number of speculative inputs to the BB
    unsigned countSpecInputs = 0;
    for (const BBArc &arc : predecessorArcs) {
      // If any of the edges in an arc is speculative, count the input arc as
      // speculative
      if (llvm::any_of(arc.edges,
                       [&](CFGEdge *p) { return speculativeEdges.count(p); }))
        countSpecInputs++;
    }

    if (countSpecInputs > 1) {
      // Potential ordering issue, add commits
      for (const BBArc &pred : predecessorArcs) {
        for (CFGEdge *edge : pred.edges) {
          // Add a Commit only in front of speculative inputs
          if (speculativeEdges.count(edge))
            placements.addCommit(*edge);
          // Here, synchronizer operations will be needed in the future
        }
      }
    }
  }

  // Now that new commits have been added, some of the already placed commits
  // might be unreachable. Hence, the path to commits is marked again and
  // unreachable commits are removed
  speculativeEdges.clear();
  markSpeculativePathsForCommits(specPos.getOwner(), placements,
                                 speculativeEdges);

  // Remove commits that cannot be reached
  llvm::DenseSet<CFGEdge *> toRemove;
  for (CFGEdge *edge : placements.getPlacements<handshake::SpecCommitOp>()) {
    if (not speculativeEdges.count(edge)) {
      toRemove.insert(edge);
    }
  }
  for (CFGEdge *edge : toRemove)
    placements.eraseCommit(*edge);
}

LogicalResult PlacementFinder::findCommitPositions() {
  OpOperand &specPos = placements.getSpeculatorPlacement();
  if (!getLogicBB(specPos.getOwner())) {
    specPos.getOwner()->emitError("Operation does not have a BB.");
    return failure();
  }

  // We need to place a commit unit before (1) an exit unit; (2) a store
  // unit; (3) a save unit if speculative tokens can reach them.
  llvm::DenseSet<Operation *> visited;
  findCommitsTraversal(visited, specPos.getOwner());

  // Additionally, if there are many BBs, two control-only tokens can
  // themselves go out of order. For this reason, additional commits need
  // to be placed in between BBs
  findCommitsBetweenBBs();

  return success();
}

//===----------------------------------------------------------------------===//
// SaveCommit Units Finder Methods
//===----------------------------------------------------------------------===//

// Traverse the speculator's BB from top to bottom (from the control merge
// until the branches) and adds save-commits in such a way that every path is
// cut by a save-commit or the speculator itself. Updates `placements`.
void PlacementFinder::findSaveCommitsTraversal(
    llvm::DenseSet<Operation *> &visited, Operation *currOp) {
  // End traversal if currOp is already in visited set
  if (auto [_, isNewOp] = visited.insert(currOp); !isNewOp)
    return;

  OpOperand &specPos = placements.getSpeculatorPlacement();
  std::optional<unsigned> specBB = getLogicBB(specPos.getOwner());

  // Verify conditions to stop the traversal on the current path
  auto stopTraversalConditions = [&](OpOperand &dstOpOperand) -> bool {
    // Stop traversal if we go outside the speculation BB
    std::optional<unsigned> succOpBB = getLogicBB(dstOpOperand.getOwner());
    if (!succOpBB || succOpBB != specBB)
      return true;

    // End traversal if the path is already cut by a commit or save-commit
    if (placements.containsCommit(dstOpOperand) ||
        placements.containsSaveCommit(dstOpOperand))
      return true;

    // End traversal if the path is already cut by the speculator
    if (&dstOpOperand == &specPos)
      return true;

    // The traversal should continue on this path
    return false;
  };

  for (Operation *succOp : currOp->getUsers()) {
    for (OpOperand &dstOpOperand : succOp->getOpOperands()) {
      // Iterate only the operands of succOp that are results from currOp
      if (dstOpOperand.get().getDefiningOp() == currOp) {
        if (stopTraversalConditions(dstOpOperand))
          continue;

        if (isa<handshake::ConditionalBranchOp>(succOp)) {
          // A SaveCommit is needed in front of the branch
          placements.addSaveCommit(dstOpOperand);
        } else {
          // Continue DFS traversal along the path
          findSaveCommitsTraversal(visited, succOp);
        }
      }
    }
  }
}

LogicalResult PlacementFinder::findSaveCommitPositions() {
  // There already exist save-commits which have been placed instead of
  // consecutive save and commit units. Here, additional save commits are
  // found
  OpOperand &specPos = placements.getSpeculatorPlacement();
  auto funcOp = specPos.getOwner()->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");

  std::optional<unsigned> specBB = getLogicBB(specPos.getOwner());
  if (!specBB) {
    specPos.getOwner()->emitError("Operation does not have a BB.");
    return failure();
  }

  // If a save-commit is placed, then for correctness, every path from entry
  // points to exit points in the Speculator BB should cross either the
  // speculator or a save-commit
  if (!placements.getPlacements<handshake::SpecSaveCommitOp>().empty()) {
    bool foundControlMerge = false;
    // Every BB starts at a control merge
    for (auto controlMergeOp : funcOp.getOps<handshake::ControlMergeOp>()) {
      if (auto mergeBB = getLogicBB(controlMergeOp);
          !mergeBB || mergeBB != specBB)
        continue;

      // Found a control merge in the speculation BB
      if (!foundControlMerge)
        foundControlMerge = true;
      else
        return controlMergeOp->emitError(
            "Found many control merges in the same BB");

      // Add commits such that all paths are cut by a save-commit or the
      // speculator
      llvm::DenseSet<Operation *> visited;
      findSaveCommitsTraversal(visited, controlMergeOp);
    }
  }

  return success();
}

LogicalResult PlacementFinder::findPlacements() {
  // Clear the data structure
  clearPlacements();

  return failure(failed(findSavePositions()) || failed(findCommitPositions()) ||
                 failed(findSaveCommitPositions()));
}
