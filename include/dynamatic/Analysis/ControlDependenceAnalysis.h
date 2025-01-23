//===- ControlDependenceAnalysis.h - Control dependence analyis *--- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions useful to analyzing the control dependencies
// between basic blocks of the CFG.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_ANALYSIS_CONTROLDEPENDENCEANALYSIS_H
#define DYNAMATIC_ANALYSIS_CONTROLDEPENDENCEANALYSIS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include <optional>

namespace dynamatic {

/// Analysis to obtain the control dependence analysis of each block in the CFG
class ControlDependenceAnalysis {

public:
  /// This structure contains all the control dependencies of a block, also
  /// separating the forward ones
  struct BlockControlDeps {
    DenseSet<Block *> allControlDeps;
    DenseSet<Block *> forwardControlDeps;
  };

  /// Store the dependencies of each block into a map
  using BlockControlDepsMap = DenseMap<Block *, BlockControlDeps>;

  // Constructor for the analysis pass
  ControlDependenceAnalysis(Operation *operation);

  // Constructor for the analysis pass
  ControlDependenceAnalysis(Region &region);

  // Given a BB, return all its control dependencies
  std::optional<DenseSet<Block *>> getBlockAllControlDeps(Block *block) const;

  // Given a BB, return all its forward control dependencies
  std::optional<DenseSet<Block *>>
  getBlockForwardControlDeps(Block *block) const;

  // Return the map of the control dependencies as
  BlockControlDepsMap getAllBlockDeps() const;

  /// Invalidation hook to keep the analysis cached across passes. Returns true
  /// if the analysis should be invalidated and fully reconstructed the next
  /// time it is queried. This is useful in case the analysis is integrated in
  /// the MLIR conversion pass
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) const {
    return !pa.isPreserved<ControlDependenceAnalysis>();
  }

  /// Print all the dependencies
  void printAllBlocksDeps() const;

private:
  BlockControlDepsMap blocksControlDeps;

  /// Get all the control dependencies of a block
  void identifyAllControlDeps(Region &region);

  // Get the forward dependencies only of a block
  void identifyForwardControlDeps(Region &region);

  /// Modify the control dependencies to include the dependencies of each
  /// dependent block too
  void addDepsOfDeps(Region &region);
};

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_CONTROLDEPENDENCEANALYSIS_H
