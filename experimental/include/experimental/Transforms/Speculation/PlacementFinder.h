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

#ifndef DYNAMATIC_TRANSFORMS_SPECULATION_PLACEMENTFINDER_H
#define DYNAMATIC_TRANSFORMS_SPECULATION_PLACEMENTFINDER_H

#include "dynamatic/Support/LLVM.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseSet.h"

namespace dynamatic {
namespace experimental {
namespace speculation {

class PlacementFinder {

public:
  /// Initializer with a SpeculationPlacements instance. Assumes that the
  /// Speculator position is set
  PlacementFinder(SpeculationPlacements &placements);

  /// Find the speculative unit positions. Mutates the internal
  /// SpeculationPlacements data structure
  LogicalResult findPlacements();

private:
  /// Mutable SpeculationPlacements data structure to hold the placements
  SpeculationPlacements &placements;

  /// Remove all existing placements except the Speculator
  void clearPlacements();

  /// Find save operations positions
  LogicalResult findSavePositions();

  /// Find commit operations positions. Uses methods findCommitsTraversal and
  /// findCommitsBetweenBBs
  LogicalResult findCommitPositions();

  /// Recursively traverse the IR in a DFS way to find the placements of commit
  /// units. See the documentation for more details:
  /// docs/Speculation/CommitUnitPlacementAlgorithm.md
  void findCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                            OpOperand &currOpOperand);

  /// Check arcs between BBs to determine if extra commits are needed to solve
  /// out-of-order tokens
  void findCommitsBetweenBBs();

  /// Find save-commit operations positions. Uses findSaveCommitsTraversal
  LogicalResult findSaveCommitPositions();

  /// DFS traversal of the speculation BB to find all SaveCommit placements
  void findSaveCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                                Operation *currOp);
};
} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_PLACEMENTFINDER_H
