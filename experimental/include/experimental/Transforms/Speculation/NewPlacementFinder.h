//===- PlacementFinder.h - Automatic speculation units finder ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the class and methods for automatic finding of speculative
// units positions.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SPECULATION_NEWPLACEMENTFINDER_H
#define DYNAMATIC_TRANSFORMS_SPECULATION_NEWPLACEMENTFINDER_H

#include "dynamatic/Support/LLVM.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>
#include <tuple>

namespace dynamatic {
namespace experimental {
namespace speculation {

struct EdgeForFlowGraph {
  int verticeTo;
  int capacity;
  int revEdgeId;
  std::optional<mlir::OpOperand *> operand;
};

constexpr int verticeTop = 0;
constexpr int verticeBottom = 1;

constexpr int capacityInf = 1000000;

class NewPlacementFinder {

public:
  /// Initializer with a SpeculationPlacements instance. Assumes that the
  /// Speculator position is set
  NewPlacementFinder(SpeculationPlacements &placements);

  /// Find the speculative unit positions. Mutates the internal
  /// SpeculationPlacements data structure
  LogicalResult findPlacements();

private:
  /// Mutable SpeculationPlacements data structure to hold the placements
  SpeculationPlacements &placements;
  OpOperand &specPos;
  unsigned specOpBB;

  std::vector<std::vector<EdgeForFlowGraph>> graphFlow;
  std::map<mlir::Operation *, int> operationToVertice;
  int verticeCount = 0;
  llvm::DenseSet<mlir::OpOperand *> directlyReachableCommits;
  llvm::DenseSet<mlir::OpOperand *> dontPlaces;

  /// Remove all existing placements except the Speculator
  void clearPlacements();

  /// Find commit operations positions. Uses methods findCommitsTraversal and
  /// findCommitsBetweenBBs
  LogicalResult findCommitPositions();

  /// DFS traversal to find the paths that need Commit units
  void findCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                            Operation *currOp);

  /// Check arcs between BBs to determine if extra commits are needed to solve
  /// out-of-order tokens
  void findCommitsBetweenBBs();

  /// Find save-commit operations positions. Uses findSaveCommitsTraversal
  LogicalResult findSaveCommitPositions();

  void constructFlowGraph();
  void constructFlowGraphRecursive(Operation *op);
  void addUndirectedEdge(int vertice1, int vertice2, int capacity,
                         std::optional<mlir::OpOperand *> operand);

  void markDontPlaces();
  void markDontPlacesRecursive(OpOperand *operand);

  void markDirectlyReachableCommits();
  void markDirectlyReachableCommitsRecursive(OpOperand &currOperand);

  int performFordFulkerson();
  int performFordFulkersonRecursive(llvm::DenseSet<int> &visited, int vertice,
                                    int flow);

  void generateSaveCommitsFromCut();
  void generateSaveCommitsFromCutRecursive(
      llvm::DenseSet<int> &visited,
      llvm::DenseSet<std::tuple<int, const EdgeForFlowGraph *>> &candidates,
      int vertice);
};
} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_NEWPLACEMENTFINDER_H
