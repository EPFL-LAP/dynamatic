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
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>

namespace dynamatic {

/// Analysis to maintain the control dependencies of basic blocks throughout the
/// IR's lifetime. Query at the beginning of Dynamatic passes using
/// `getAnalysis<ControlDependenceAnalysis>()` and cache it to avoid
/// recomputations for further passes using
/// `markAnalysesPreserved<ControlDependenceAnalysis>()` at the end.
template <typename FunctionType>
class ControlDependenceAnalysis {
public:
  /// Constructor called automatically by
  /// `getAnalysis<ControlDependenceAnalysis>()` if the analysis is not already
  /// cached.
  ControlDependenceAnalysis(Operation *operation) {
    // type-cast it into FunctionType
    FunctionType funcOp = dyn_cast<FunctionType>(operation);
    if (funcOp) {
      // if the operation is a function, then identify all the control
      // dependencies among its BBs
      identifyAllControlDeps(funcOp);
    } else {
      // report an error indicating that the analysis is instantiated over
      // an inappropriate operation
      llvm::errs() << "ControlDependenceAnalysis is instantiated over an "
                      "operation that is not FuncOp!\n";
    }
  };

  // given a BB within that function, return all the BBs the block is control
  // dependant on.
  std::optional<DenseSet<Block *>> getBlockAllControlDeps(Block *block) const;

  // given a BB within that function, return all the BBs the bloc
  // is control dependant on, without taking into account backward dependencies
  // (i.e. excluding loop exits)
  std::optional<DenseSet<Block *>>
  getBlockForwardControlDeps(Block *block) const;

  /// Invalidation hook to keep the analysis cached across passes. Returns true
  /// if the analysis should be invalidated and fully reconstructed the next
  /// time it is queried. This is useful in case the analysis is integrated in
  /// the MLIR conversion pass
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) const {
    return !pa.isPreserved<ControlDependenceAnalysis>();
  }

  void printAllBlocksDeps() const;

private:
  // Each block has a structure of type `BlockControlDeps` containing the list
  // of BBs it is control dependent on, both in the forward and backward
  // direction.
  struct BlockControlDeps {
    DenseSet<Block *> allControlDeps;
    DenseSet<Block *> forwardControlDeps;
  };
  using BlockControlDepsMap = DenseMap<Block *, BlockControlDeps>;

  // Store the list of dependencies for each block
  BlockControlDepsMap blocksControlDeps;

  // Fill the `allControlDeps` field of the entry in `funcBlocksControlDeps`
  // corresponding to the input `funcOp`
  void identifyAllControlDeps(FunctionType &funcOp);

  // Fill the `forwardControlDeps` field of the entry in `funcBlocksControlDeps`
  // corresponding to the input `funcOp`
  void identifyForwardControlDeps(FunctionType &funcOp);

  // Given a start block and en end block withing a function region, return all
  // the post dominator trees
  void enumeratePathsInPostDomTree(
      Block *startBlock, Block *endBlock, Region *funcReg,
      llvm::DominatorTreeBase<Block, true> *postDomTree,
      SmallVector<SmallVector<mlir::DominanceInfoNode *>> *traversedNodes);

  // Helper recursive function called inside `enumeratePathsInPostDomTree`
  void enumeratePathsInPostDomTreeUtil(
      mlir::DominanceInfoNode *startNode, mlir::DominanceInfoNode *endNode,
      DenseMap<mlir::DominanceInfoNode *, bool> isVisited,
      SmallVector<mlir::DominanceInfoNode *> path, int pathIndex,
      SmallVector<SmallVector<mlir::DominanceInfoNode *>> *traversedNodes);

  // adjusts the dependencies of each block to include nested dependencies
  // (i.e., the dependencies of its depenendencies)
  void addDepsOfDeps(FunctionType &funcOp,
                     BlockControlDepsMap &blockControlDepsMap);
};

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_CONTROLDEPENDENCEANALYSIS_H
