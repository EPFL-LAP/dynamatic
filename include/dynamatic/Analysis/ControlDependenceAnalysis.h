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
      identifyAllControlDeps(funcOp);
    } else {
      ModuleOp modOp = dyn_cast<ModuleOp>(operation);
      if (modOp) {
        auto funcOps = modOp.getOps<mlir::func::FuncOp>();
        // call those in a loop and create a big structure
        for (mlir::func::FuncOp funcOp : llvm::make_early_inc_range(funcOps))
          identifyAllControlDeps(funcOp);
      } else {
        // report an error indicating that the anaylsis is instantiated over
        // an inappropriate operation
        llvm::errs() << "ControlDependenceAnalysis is instantiated over an "
                        "operation that is not FuncOp or ModuleOp!\n";
      }
    }
  };

  // All public functions are accessible from outside of the pass and require
  // the outside to specify the index of the funcOp of interest. If the outside
  // has a func::FuncOp rather than a ModuleOp, they should just pass a 0

  // return all BBs that the block in the argument, belonging to the funcOp of
  // the passed idx, is control dependent on
  void calculateBlockControlDeps(Block *block, int funcOp_idx,
                                 SmallVector<Block *> &returned_control_deps);

  // return only forward dependencies (i.e., excluding loop exits) that the
  // block in the argument, belonging to the funcOp of the passed idx, is
  // control dependent on
  void calculateBlockForwardControlDeps(
      Block *block, int funcOp_idx,
      SmallVector<Block *> &returned_forward_control_deps);

  /// Invalidation hook to keep the analysis cached across passes. Returns true
  /// if the analysis should be invalidated and fully reconstructed the next
  /// time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<ControlDependenceAnalysis>();
  }

private:
  SmallVector<DenseMap<Block *, SmallVector<Block *, 6>>, 6>
      all_control_deps_maps;

  // For every func::FuncOp, contains only the forward control dependencies,
  // excluding the loop exit conditions
  SmallVector<DenseMap<Block *, SmallVector<Block *, 6>>, 6>
      forward_control_deps_maps;

  // Simply fill one entry of all_control_deps_maps, corresponding to one funcOp
  void identifyAllControlDeps(mlir::func::FuncOp &funcOp);

  // helper function called inside identifyAllControlDeps
  void addDepsFromPostDomTree();

  // Simply extract one entry of forward_control_deps_maps from the
  // all_control_deps_maps, corresponding to one funcOp
  void identifyForwardControlDeps(mlir::func::FuncOp &funcOp);

  // Returns all postDominator tree nodes between start_node and end_node in the
  // postDominator tree
  void enumeratePathsInPostDomTree(
      Block *start_block, Block *end_block, Region *funcReg,
      llvm::DominatorTreeBase<Block, true> *postDomTree,
      SmallVector<SmallVector<mlir::DominanceInfoNode *, 6>, 6>
          *traversed_nodes);

  // recursive function called inside enumeratePathsInPostDomTree
  void enumeratePathsInPostDomTreeUtil(
      mlir::DominanceInfoNode *start_node, mlir::DominanceInfoNode *end_node,
      DenseMap<mlir::DominanceInfoNode *, bool> is_visited,
      SmallVector<mlir::DominanceInfoNode *, 6> path, int path_index,
      SmallVector<SmallVector<mlir::DominanceInfoNode *, 6>, 6>
          *traversed_nodes);

  // adjusts the dependencies of each block to include nested dependencies
  // (i.e., the dependencies of its depenendencies)
  void addDepsOfDeps(mlir::func::FuncOp &funcOp,
                     DenseMap<Block *, SmallVector<Block *>> &control_deps_map);
};

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_CONTROLDEPENDENCEANALYSIS_H
