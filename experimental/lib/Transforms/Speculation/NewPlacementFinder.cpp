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

#include "experimental/Transforms/Speculation/NewPlacementFinder.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include <iostream>
#include <optional>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculation;

NewPlacementFinder::NewPlacementFinder(SpeculationPlacements &placements)
    : placements(placements), specPos(placements.getSpeculatorPlacement()) {
  Operation *specOp = specPos.getOwner();
  assert(specOp && "Speculator position is undefined");
  if (auto bb = getLogicBB(specOp)) {
    specOpBB = bb.value();
  } else {
    specOp->emitError("Operation does not have a BB.");
  }
}

void NewPlacementFinder::clearPlacements() {
  OpOperand &specPos = placements.getSpeculatorPlacement();
  auto buffers = this->placements.getPlacements<handshake::BufferOp>();
  this->placements = SpeculationPlacements(specPos, buffers);
}

// Recursively traverse the IR in a DFS way to find the placements of Commit
// units. A commit unit before (1) an exit unit; (2) a store unit; (3) a save
// unit if speculative tokens can reach them. Updates the `placements`
void NewPlacementFinder::findCommitsTraversal(
    llvm::DenseSet<Operation *> &visited, Operation *currOp) {
  // End traversal if currOp is already in visited set
  if (auto [_, isNewOp] = visited.insert(currOp); !isNewOp)
    return;

  for (OpResult res : currOp->getResults()) {
    for (OpOperand &dstOpOperand : res.getUses()) {
      Operation *succOp = dstOpOperand.getOwner();
      // if (placements.containsSave(dstOpOperand)) {
      //   // A Commit is needed in front of Save Operations. To allow for
      //   // multiple loop speculation, SaveCommit units are used instead of
      //   // consecutive Commit-Save units.
      //   placements.addSaveCommit(dstOpOperand);
      //   placements.eraseSave(dstOpOperand);
      // } else if (isa<handshake::StoreOpInterface>(succOp)) {
      if (isa<handshake::StoreOp>(succOp)) {
        // A commit is needed in front of memory operations
        placements.addCommit(dstOpOperand);
      } else if (isa<handshake::EndOp>(succOp)) {
        // A commit is needed in front of the end/exit operation
        placements.addCommit(dstOpOperand);
      } else if (isa<handshake::MemoryControllerOp>(succOp)) {
        if (dstOpOperand.getOperandNumber() == 2 &&
            !isa<handshake::LoadOp>(currOp)) {
          // A commit is needed in front of the memory controller
          // On the operand indicating the number of stores
          placements.addCommit(dstOpOperand);
        } else if (dstOpOperand.get().getType().isa<handshake::ControlType>()) {
          // End signal
          placements.addCommit(dstOpOperand);
        }
        // Exceptionally stop the traversal
        continue;
      } else {
        findCommitsTraversal(visited, succOp);
      }
    }
  }
}

LogicalResult NewPlacementFinder::findCommitPositions() {
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

namespace {
using CFGEdge = OpOperand;
}

// DFS traversal to mark all operations that lead to Commit units
// The set markedPaths is passed by reference and is updated with
// the OpPlacements (pair value-operation) that are traversed
static void
markSpeculativePathsForCommits(Operation *currOp,
                               SpeculationPlacements &placements,
                               llvm::DenseSet<CFGEdge *> &markedEdges) {
  for (OpResult res : currOp->getResults()) {
    for (OpOperand &edge : res.getUses()) {
      if (!markedEdges.count(&edge)) {
        markedEdges.insert(&edge);
        // Stop traversal if a commit is reached
        if (!placements.containsCommit(edge))
          markSpeculativePathsForCommits(edge.getOwner(), placements,
                                         markedEdges);
      }
    }
  }
}

// Find the placements of Commit units in between BBs, that are needed to
// avoid two control-only tokens going out of order. Updates the `placements`
void NewPlacementFinder::findCommitsBetweenBBs() {
  auto funcOp = specPos.getOwner()->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");

  // Whenever a BB has two speculative inputs, commit units are needed to
  // avoid tokens going out-of-order. First, the block predecessor arcs are
  // found
  BBtoArcsMap bbToPredecessorArcs = getBBPredecessorArcs(funcOp);

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
      int placeIndex = 0;
      int i = 0;
      // Potential ordering issue, add commits
      for (const BBArc &pred : predecessorArcs) {
        if (i == placeIndex) {
          for (CFGEdge *edge : pred.edges) {
            // Add a Commit only in front of speculative inputs
            if (speculativeEdges.count(edge))
              placements.addCommit(*edge);
            // Here, synchronizer operations will be needed in the future
          }
        }
        i++;
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
    if (!speculativeEdges.count(edge)) {
      toRemove.insert(edge);
    }
  }
  for (CFGEdge *edge : toRemove)
    placements.eraseCommit(*edge);
}

LogicalResult NewPlacementFinder::findSaveCommitPositions() {
  constructFlowGraph();
  performFordFulkerson();
  generateSaveCommitsFromCut();
  return success();
}

static std::optional<handshake::ControlMergeOp>
findControlMergeOp(handshake::FuncOp &funcOp, unsigned bb) {
  for (auto mergeOp : funcOp.getOps<handshake::ControlMergeOp>()) {
    if (auto mergeBB = getLogicBB(mergeOp); mergeBB && mergeBB == bb)
      return mergeOp;
  }
  return std::nullopt;
}

void NewPlacementFinder::constructFlowGraph() {
  auto funcOp = specPos.getOwner()->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "op should have parent function");

  // Here, we assume 1-BB loop
  // cmerge can be found
  auto cmergeOp = findControlMergeOp(funcOp, specOpBB);
  if (!cmergeOp) {
    specPos.getOwner()->emitError("Control merge not found in the BB.");
    return;
  }

  graphFlow.clear();
  graphFlow.resize(100); // TODO
  operationToVertice.clear();
  verticeCount = 2;
  dontPlaces.clear();

  markDirectlyReachableCommits();

  markDontPlaces();

  // todo: resize

  constructFlowGraphRecursive(cmergeOp.value());
}

void NewPlacementFinder::constructFlowGraphRecursive(Operation *op) {
  // assume that the operation is visited only once
  int verticeId = verticeCount;
  operationToVertice[op] = verticeId;
  verticeCount++;
  // todo: resize
  if (isa<handshake::ControlMergeOp>(op) || isa<handshake::MuxOp>(op)) {
    // connect with the verticeTop with infinite capacity
    addUndirectedEdge(verticeTop, verticeId, capacityInf, std::nullopt);
  } else if (isa<handshake::ConditionalBranchOp>(op)) {
    // connect with the verticeBottom with infinite capacity
    addUndirectedEdge(verticeBottom, verticeId, capacityInf, std::nullopt);
    // todo: return
  }
  for (auto result : op->getResults()) {
    for (auto &use : result.getUses()) {
      auto *owner = use.getOwner();

      if (isBackedge(result, owner)) {
        continue;
      }

      if (auto bb = getLogicBB(owner); !bb || bb.value() != specOpBB) {
        continue;
      }

      if (placements.containsCommit(use)) {
        // out of speculative region
        continue;
      }

      int nextVerticeId;
      bool visitNextOp;
      if (operationToVertice.count(owner)) {
        nextVerticeId = operationToVertice[owner];
        visitNextOp = false;
      } else {
        nextVerticeId = verticeCount;
        visitNextOp = true;
      }

      int capacity;
      if (dontPlaces.contains(&use)) {
        capacity = capacityInf;
      } else {
        capacity = 1;
      }
      addUndirectedEdge(verticeId, nextVerticeId, capacity, &use);

      if (&use == &specPos) {
        addUndirectedEdge(verticeTop, verticeId, capacityInf, std::nullopt);
        addUndirectedEdge(verticeBottom, nextVerticeId, capacityInf,
                          std::nullopt);
      }

      if (visitNextOp) {
        constructFlowGraphRecursive(owner);
      }
    }
  }
}

void NewPlacementFinder::addUndirectedEdge(
    int vertice1, int vertice2, int capacity,
    std::optional<mlir::OpOperand *> operand) {
  EdgeForFlowGraph edge;
  edge.verticeTo = vertice2;
  edge.capacity = capacity;
  edge.revEdgeId = graphFlow[vertice2].size();
  edge.operand = operand;
  graphFlow[vertice1].emplace_back(edge);
  EdgeForFlowGraph revEdge;
  revEdge.verticeTo = vertice1;
  revEdge.capacity = capacity;
  revEdge.revEdgeId = graphFlow[vertice1].size() - 1;
  revEdge.operand = operand;
  graphFlow[vertice2].emplace_back(revEdge);
}

void NewPlacementFinder::markDontPlaces() {
  std::cerr << "number of commits: "
            << placements.getPlacements<handshake::SpecCommitOp>().size()
            << "\n";
  for (auto *commitOperand :
       placements.getPlacements<handshake::SpecCommitOp>()) {
    if (directlyReachableCommits.contains(commitOperand)) {
      continue;
    }
    std::cerr << "cantReachCommit: \n";
    commitOperand->getOwner()->dump();
    // traverse the commit operand upto mux or cmerge
    markDontPlacesRecursive(commitOperand);
  }
}

void NewPlacementFinder::markDontPlacesRecursive(OpOperand *operand) {
  if (dontPlaces.contains(operand))
    return;
  // is this happen?
  // if (auto bb = getLogicBB(op); !bb || bb.value() != specOpBB)
  //   return;

  std::cerr << "dontPlace: \n";
  operand->getOwner()->dump();
  std::cerr << operand->getOperandNumber() << "\n";
  dontPlaces.insert(operand);

  Operation *op = operand->get().getDefiningOp();
  if (!op)
    return;
  if (isa<handshake::MuxOp>(op))
    return;
  if (isa<handshake::ControlMergeOp>(op))
    return;

  for (auto &operand : op->getOpOperands()) {
    markDontPlacesRecursive(&operand);
  }
}

void NewPlacementFinder::markDirectlyReachableCommits() {
  markDirectlyReachableCommitsRecursive(specPos);
}

void NewPlacementFinder::markDirectlyReachableCommitsRecursive(
    OpOperand &currOperand) {
  for (auto result : currOperand.getOwner()->getResults()) {
    for (auto &use : result.getUses()) {
      if (isBackedge(result, use.getOwner())) {
        continue;
      }
      if (placements.containsCommit(use)) {
        std::cerr << "directlyReachableCommits: \n";
        use.getOwner()->dump();
        directlyReachableCommits.insert(&use);
        // No commits after the commit
        continue;
      }
      markDirectlyReachableCommitsRecursive(use);
    }
  }
}

int NewPlacementFinder::performFordFulkerson() {
  int flow = 0;
  llvm::DenseSet<int> visited;
  while (true) {
    visited.clear();
    int newFlow =
        performFordFulkersonRecursive(visited, verticeBottom, capacityInf);
    if (newFlow == 0) {
      std::cerr << "Flow size: " << flow << std::endl;
      return flow;
    }
    flow += newFlow;
  }
}

int NewPlacementFinder::performFordFulkersonRecursive(
    llvm::DenseSet<int> &visited, int vertice, int flow) {
  if (vertice == verticeTop) {
    return flow;
  }
  visited.insert(vertice);
  for (auto &edge : graphFlow[vertice]) {
    if (!visited.contains(edge.verticeTo) && edge.capacity > 0) {
      int newFlow = performFordFulkersonRecursive(
          visited, edge.verticeTo, std::min(flow, edge.capacity));
      if (newFlow > 0) {
        edge.capacity -= newFlow;
        graphFlow[edge.verticeTo][edge.revEdgeId].capacity += newFlow;
        return newFlow;
      }
    }
  }
  return 0;
}

void NewPlacementFinder::generateSaveCommitsFromCut() {
  llvm::DenseSet<int> visited;
  llvm::DenseSet<std::tuple<int, const EdgeForFlowGraph *>> candidates;
  generateSaveCommitsFromCutRecursive(visited, candidates, verticeBottom);
  for (auto candidate : candidates) {
    auto [from, edge] = candidate;
    if (visited.contains(from) && !visited.contains(edge->verticeTo)) {
      if (edge->operand.has_value()) {
        if (edge->operand.value() == &specPos) {
          // Speculator
          continue;
        }
        std::cerr << "SaveCommit: \n";
        edge->operand.value()->getOwner()->dump();
        std::cerr << edge->operand.value()->getOperandNumber() << "\n";
        placements.addSaveCommit(*(edge->operand.value()));
      } else {
        // ERROR
      }
    }
  }
}

void NewPlacementFinder::generateSaveCommitsFromCutRecursive(
    llvm::DenseSet<int> &visited,
    llvm::DenseSet<std::tuple<int, const EdgeForFlowGraph *>> &candidates,
    int vertice) {
  visited.insert(vertice);
  for (auto &edge : graphFlow[vertice]) {
    if (edge.capacity == 0) { // <= ?
      candidates.insert({vertice, &edge});
    } else {
      if (!visited.contains(edge.verticeTo)) {
        generateSaveCommitsFromCutRecursive(visited, candidates,
                                            edge.verticeTo);
      }
    }
  }
}

LogicalResult NewPlacementFinder::findPlacements() {
  // Clear the data structure
  clearPlacements();

  return failure(failed(findCommitPositions()) ||
                 failed(findSaveCommitPositions()));
}
