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

namespace dynamatic {

/// Analysis to maintain the control dependencies of basic blocks throughout the
/// IR's lifetime. Query at the beginning of Dynamatic passes using
/// `getAnalysis<ControlDependenceAnalysis>()` and cache it to avoid
/// recomputations for further passes using
/// `markAnalysesPreserved<ControlDependenceAnalysis>()` at the end.
///
class ControlDependenceAnalysis {
public:
  /// Constructor called automatically by
  /// `getAnalysis<ControlDependenceAnalysis>()` if the analysis is not already
  /// cached.
  ControlDependenceAnalysis(Operation *operation) {
    // type-cast it into mlir::func::FuncOp
    mlir::func::FuncOp funcOp = dyn_cast<mlir::func::FuncOp>(operation);
    if (funcOp) {
      // if the operation is a function, then identify all the control
      // dependencies among its BBs
      identifyAllControlDeps(funcOp);
    } else {
      ModuleOp modOp = dyn_cast<ModuleOp>(operation);
      if (modOp) {
        // If the operation is a module, then analyze all the control
        // dependencies among the functions in contains
        auto funcOps = modOp.getOps<mlir::func::FuncOp>();
        for (mlir::func::FuncOp funcOp : llvm::make_early_inc_range(funcOps))
          identifyAllControlDeps(funcOp);
      } else {
        // report an error indicating that the analysis is instantiated over
        // an inappropriate operation
        llvm::errs() << "ControlDependenceAnalysis is instantiated over an "
                        "operation that is not FuncOp or ModuleOp!\n";
      }
    }
  };

  // given a FuncOp and a BB within that function, return all the BBs the bloc
  // is control dependant on.
  LogicalResult getBlockAllControlDeps(Block *block, mlir::func::FuncOp &funcOp,
                                       DenseSet<Block *> &allControlDeps) const;

  // given a FuncOp and a BB within that function, return all the BBs the bloc
  // is control dependant on, without taking into account backward dependencies
  // (i.e. excluding loop exits)
  LogicalResult
  getBlockForwardControlDeps(Block *block, mlir::func::FuncOp &funcOp,
                             DenseSet<Block *> &forwardControlDeps) const;

  /// Invalidation hook to keep the analysis cached across passes. Returns true
  /// if the analysis should be invalidated and fully reconstructed the next
  /// time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) const {
    return !pa.isPreserved<ControlDependenceAnalysis>();
  }

  LogicalResult printAllBlocksDeps(mlir::func::FuncOp &) const;

private:
  // Each block has a structure of type `BlockControlDeps` containing the list
  // of BBs it is control dependent on, both in the forward and backward
  // direction.
  struct BlockControlDeps {
    DenseSet<Block *> allControlDeps;
    DenseSet<Block *> forwardControlDeps;
  };
  using BlockControlDepsMap = DenseMap<Block *, BlockControlDeps>;

  // For each FuncOp, `funcBlocksControlDeps` stores a `BlockControlDeps` for
  // each block.
  DenseMap<mlir::func::FuncOp, BlockControlDepsMap> funcBlocksControlDeps;

  // Fill the `allControlDeps` field of the entry in `funcBlocksControlDeps`
  // corresponding to the input `funcOp`
  void identifyAllControlDeps(mlir::func::FuncOp &funcOp);

  // Fill the `forwardControlDeps` field of the entry in `funcBlocksControlDeps`
  // corresponding to the input `funcOp`
  void identifyForwardControlDeps(mlir::func::FuncOp &funcOp);

  // Given a start block and en end block withing a function region, return all
  // the post dominator trees
  void enumeratePathsInPostDomTree(
      Block *startBlock, Block *endBlock, Region *funcReg,
      llvm::DominatorTreeBase<Block, true> *postDomTree,
      SmallVector<SmallVector<mlir::DominanceInfoNode *, 6>, 6>
          *traversedNodes);

  // Helper recursive function called inside `enumeratePathsInPostDomTree`
  void enumeratePathsInPostDomTreeUtil(
      mlir::DominanceInfoNode *startNode, mlir::DominanceInfoNode *endNode,
      DenseMap<mlir::DominanceInfoNode *, bool> isVisited,
      SmallVector<mlir::DominanceInfoNode *, 6> path, int pathIndex,
      SmallVector<SmallVector<mlir::DominanceInfoNode *, 6>, 6>
          *traversedNodes);

  // adjusts the dependencies of each block to include nested dependencies
  // (i.e., the dependencies of its depenendencies)
  void addDepsOfDeps(mlir::func::FuncOp &funcOp,
                     BlockControlDepsMap &blockControlDepsMap);
};

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_CONTROLDEPENDENCEANALYSIS_H
