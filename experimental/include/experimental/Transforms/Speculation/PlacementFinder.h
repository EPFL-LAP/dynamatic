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
  LogicalResult findSaves();

  /// Identifies positions of regular commits that prevent side effects.
  LogicalResult findRegularCommits();

  /// Recursively traverse the IR in a DFS way to find the placements of commit
  /// units.
  LogicalResult
  findRegularCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                              OpOperand &currOpOperand);

  /// Additional commits prevent token reordering in multiple-BB cases.
  /// Check arcs between BBs to determine if extra commits are needed to
  /// solve out-of-order tokens
  LogicalResult findCommitsBetweenBBs();

  /// Identifies save-commit positions.
  LogicalResult findSaveCommits();

  /// DFS traversal of the speculation BB to find all SaveCommit placements
  LogicalResult findSaveCommitsTraversal(llvm::DenseSet<Operation *> &visited,
                                         Operation *currOp);
};
} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_PLACEMENTFINDER_H
