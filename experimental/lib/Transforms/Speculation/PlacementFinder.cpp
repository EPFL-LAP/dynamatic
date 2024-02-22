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
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculation;

PlacementFinder::PlacementFinder(SpeculationPlacements &placements)
    : placements(placements) {
  OpPlacement specPos = placements.getSpeculatorPlacement();
  assert(specPos.dstOp != nullptr && "Speculator position is undefined");
}

void PlacementFinder::clearPlacements() {
  OpPlacement specPos = placements.getSpeculatorPlacement();
  this->placements = SpeculationPlacements(specPos.srcOpResult, specPos.dstOp);
}

//===----------------------------------------------------------------------===//
// Save Units Finder Methods
//===----------------------------------------------------------------------===//

// Recursively traverse the IR until reaching branches and store visited values
static void markSpeculativePaths(Operation *currOp,
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
      markSpeculativePaths(succOp, specValues);
    }
  }
}

// Save units are needed where speculative tokens can interact with
// non-speculative tokens
LogicalResult PlacementFinder::findSavePositions() {
  OpPlacement specPos = placements.getSpeculatorPlacement();
  handshake::FuncOp funcOp =
      specPos.dstOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");
  auto handshakeBlocks = getLogicBBs(funcOp);

  // Mark all values that are speculative in the speculation BB
  llvm::DenseSet<Value> specValues;
  specValues.insert(specPos.srcOpResult);
  markSpeculativePaths(specPos.dstOp, specValues);

  // Iterate all operations in the speculation BB
  unsigned bb = getLogicBB(specPos.dstOp).value();
  for (Operation *blockOp : handshakeBlocks.blocks.lookup(bb)) {
    // Create a save if an operation has both spec and non-spec operands
    bool hasNonSpecInput = false;
    bool hasSpecInput = false;
    for (Value operand : blockOp->getOperands()) {
      if (specValues.count(operand))
        hasSpecInput = true;
      else
        hasNonSpecInput = true;
    }

    if (hasSpecInput and hasNonSpecInput) {
      for (Value operand : blockOp->getOperands()) {
        // Create a Save for every non-speculative operand
        if (not specValues.count(operand)) {
          // No save needed in front of Source Operations
          if (isa<handshake::SourceOp>(operand.getDefiningOp()))
            continue;

          placements.addSave(operand, blockOp);
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Commit Units Finder Methods
//===----------------------------------------------------------------------===//

void PlacementFinder::findCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                                           Operation *currOp) {
  // End traversal if currOp is already in visited set
  if (auto [_, isNewOp] = visited.insert(currOp); !isNewOp)
    return;
  for (Value res : currOp->getResults()) {
    for (Operation *succOp : res.getUsers()) {
      if (placements.containsSave(res, succOp)) {
        // A Commit is needed in front of Save Operations. This will be later
        // converted into a SaveCommit for multiple loop speculation.
        placements.addCommit(res, succOp);
      } else if (isa<handshake::LSQOp, handshake::MemoryControllerOp,
                     handshake::MCStoreOp, handshake::LSQStoreOp,
                     handshake::MCLoadOp, handshake::LSQLoadOp>(succOp)) {
        // A commit is needed in front of memory operations
        placements.addCommit(res, succOp);
      } else if (isa<handshake::EndOp>(succOp)) {
        // A commit is needed in front of the end/exit operation
        placements.addCommit(res, succOp);
      } else {
        findCommitsTraversal(visited, succOp);
      }
    }
  }
}

struct endpointComparator {
  bool operator()(const BBEndpoints &a, const BBEndpoints &b) const {
    if (a.srcBB != b.srcBB)
      return a.srcBB < b.srcBB;
    return a.dstBB < b.dstBB;
  }
};
using BBEndpointsMap =
    std::map<BBEndpoints, std::vector<OpPlacement>, endpointComparator>;

// Data structure to hold all arcs leading to a single BB predecessor
// Note: srcBB and dstBB can be equal when the arcs are Backedges
struct BlockPredecessor {
  unsigned srcBB;
  unsigned dstBB;
  std::vector<OpPlacement> arcs;
};
using BBPredecessorMap = std::map<unsigned, std::vector<BlockPredecessor>>;

// Get a map from BBs to the list of its BlockPredecessors
static BBPredecessorMap getBlockPredecessors(handshake::FuncOp funcOp) {
  // Traverse all operations to find arcs between BBs (and self-arcs)
  BBEndpointsMap endpointArcs;
  funcOp->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      BBEndpoints endpoints;
      if (isBackedge(operand, op, &endpoints) or
          endpoints.srcBB != endpoints.dstBB) {
        OpPlacement arc = {operand, op};
        endpointArcs[endpoints].push_back(arc);
      }
    }
  });

  // Join all predecessors of a BB
  BBPredecessorMap predecessors;
  for (const auto &[endpoints, arcs] : endpointArcs) {
    BlockPredecessor predecessor;
    predecessor.srcBB = endpoints.srcBB;
    predecessor.dstBB = endpoints.dstBB;
    predecessor.arcs = arcs;
    predecessors[endpoints.dstBB].push_back(predecessor);
  }

  return predecessors;
}

// DFS traversal to mark all operations that lead to Commit units
static void markSpeculativePaths(Operation *currOp,
                                 SpeculationPlacements &placements,
                                 PlacementList &markedPaths) {
  for (Value res : currOp->getResults()) {
    for (Operation *succOp : res.getUsers()) {
      OpPlacement arc = {res, succOp};
      if (not markedPaths.count(arc)) {
        markedPaths.insert(arc);
        // Stop traversal if a commit is reached
        if (not placements.containsCommit(res, succOp))
          markSpeculativePaths(succOp, placements, markedPaths);
      }
    }
  }
}

void PlacementFinder::findCommitsBetweenBBs() {
  OpPlacement specPos = placements.getSpeculatorPlacement();
  auto funcOp = specPos.dstOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");

  // Whenever a BB has two speculative inputs, commit units are needed to avoid
  // tokens going out-of-order. First, the block predecessor arcs are found
  BBPredecessorMap predecessors = getBlockPredecessors(funcOp);
  PlacementList markedPaths;
  markSpeculativePaths(specPos.dstOp, placements, markedPaths);
  // Iterate all BBs to check if commits are needed
  for (const auto &[bb, blockPredecessors] : predecessors) {
    // Count number of speculative inputs to the BB
    unsigned countSpecInputs = 0;
    for (BlockPredecessor pred : blockPredecessors) {
      if (llvm::any_of(pred.arcs,
                       [&](OpPlacement p) { return markedPaths.count(p); }))
        countSpecInputs++;
    }

    if (countSpecInputs > 1) {
      // Potential ordering issue, add commits
      for (BlockPredecessor pred : blockPredecessors) {
        for (OpPlacement arc : pred.arcs) {
          placements.addCommit(arc.srcOpResult, arc.dstOp);
          // Here, synchronizer operations will be needed in the future
        }
      }
    }
  }

  // Now that new commits have been added, some of the already placed commits
  // might be unreachable. Hence, the path to commits is marked again and
  // unreachable commits are removed
  markedPaths.clear();
  markSpeculativePaths(specPos.dstOp, placements, markedPaths);

  // Remove commits that cannot be reached
  PlacementList toRemove;
  for (OpPlacement arc : placements.getPlacements<handshake::SpecCommitOp>()) {
    if (not markedPaths.count(arc)) {
      toRemove.insert({arc.srcOpResult, arc.dstOp});
    }
  }
  for (OpPlacement p : toRemove) {
    placements.eraseCommit(p.srcOpResult, p.dstOp);
  }
}

LogicalResult PlacementFinder::findCommitPositions() {
  OpPlacement specPos = placements.getSpeculatorPlacement();

  // We need to place a commit unit before (1) an exit unit; (2) a store
  // unit; (3) a save unit if speculative tokens can reach them.
  llvm::DenseSet<Operation *> visited;
  findCommitsTraversal(visited, specPos.dstOp);

  // Additionally, if there are many BBs, two control-only tokens can
  // themselves go out of order. For this reason, additional commits need
  // to be placed in between BBs
  findCommitsBetweenBBs();

  return success();
}

//===----------------------------------------------------------------------===//
// SaveCommit Units Finder Methods
//===----------------------------------------------------------------------===//

void PlacementFinder::findSaveCommitsTraversal(
    llvm::DenseSet<Operation *> &visited, Operation *currOp) {
  // End traversal if currOp is already in visited set
  if (auto [_, isNewOp] = visited.insert(currOp); !isNewOp)
    return;

  OpPlacement specPos = placements.getSpeculatorPlacement();
  auto specBB = getLogicBB(specPos.dstOp);

  for (Value res : currOp->getResults()) {
    for (Operation *succOp : res.getUsers()) {
      if (auto succOpBB = getLogicBB(succOp); !succOpBB || succOpBB != specBB) {
        // Stop traversal if we go outside the speculation BB
        continue;
      } else if (placements.containsCommit(res, succOp) or
                 placements.containsSaveCommit(res, succOp)) {
        // End traversal on this path, as it already is cut by a commit or sc
        continue;
      } else if (OpPlacement specPos = placements.getSpeculatorPlacement();
                 specPos.srcOpResult == res and specPos.dstOp == succOp) {
        // End traversal on this path, as it already is cut by the speculator
        continue;
      } else if (isa<handshake::ConditionalBranchOp>(succOp)) {
        // A SaveCommit is needed in front of the branch
        placements.addSaveCommit(res, succOp);
      } else {
        // Continue DFS traversal along the path
        findSaveCommitsTraversal(visited, succOp);
      }
    }
  }
}

LogicalResult PlacementFinder::findSaveCommitPositions() {
  // Merge consecutive save and commit units into save-commits
  placements.mergeSaveCommits();

  // Find additional save commits
  OpPlacement specPos = placements.getSpeculatorPlacement();
  auto funcOp = specPos.dstOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");

  auto specBB = getLogicBB(specPos.dstOp);
  if (not specBB) {
    specPos.dstOp->emitError("Operation does not have a BB.");
    return failure();
  }

  // Every path from entry points to exit points in the Speculator BB should
  // cross either the speculator or a save-commit
  if (not placements.getPlacements<handshake::SpecSaveCommitOp>().empty()) {
    bool foundControlMerge = false;
    // Every BB starts at a control merge
    for (auto controlMergeOp : funcOp.getOps<handshake::ControlMergeOp>()) {
      if (auto mergeBB = getLogicBB(controlMergeOp);
          !mergeBB || mergeBB != specBB)
        continue;

      // Found a control merge in the speculation BB
      if (not foundControlMerge) {
        foundControlMerge = true;
      } else {
        controlMergeOp->emitError("Found many control merges in the same BB");
        return failure();
      }

      // Add commits such that all paths are cut by a save-commit or the
      // speculator.
      llvm::DenseSet<Operation *> visited;
      findSaveCommitsTraversal(visited, controlMergeOp);
    }
  }

  return success();
}

LogicalResult PlacementFinder::findPlacements() {
  // Clear the data structure
  clearPlacements();

  if (failed(findSavePositions()))
    return failure();

  if (failed(findCommitPositions()))
    return failure();

  if (failed(findSaveCommitPositions()))
    return failure();

  return success();
}
