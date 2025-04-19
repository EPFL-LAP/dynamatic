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

  /// Find save operations positions
  LogicalResult findSavePositions();

  /// Determines the placement of commit units that are reachable directly from
  /// the speculator (i.e., without passing through any save-commits). Also
  /// updates the placement of save units to that of save-commits when
  /// encountered during traversal.
  LogicalResult findRegularCommitsAndSCs();

  /// Recursively traverse the IR in a DFS way to find the placements of commit
  /// units. See the documentation for more details:
  /// docs/Speculation/CommitUnitPlacementAlgorithm.md
  void findCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                            OpOperand &currOpOperand);

  /// Additional commits are needed to avoid out-of-order tokens in multiple-BB
  /// cases.
  /// Check arcs between BBs to determine if extra commits are needed to
  /// solve out-of-order tokens
  LogicalResult findCommitsBetweenBBs();

  /// Identifies additional commit positions that are reachable only through
  /// certain save-commit units.
  LogicalResult findRegularCommitsFromSCs();

  /// Identifies additional save-commit positions, referred to as "snapshots".
  LogicalResult findSnapshotSCs();

  /// DFS traversal of the speculation BB to find all SaveCommit placements
  LogicalResult findSaveCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                                         Operation *currOp);
};
} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_PLACEMENTFINDER_H
