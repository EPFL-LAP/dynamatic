//===- ControlDependenceAnalysis.h - Control dependence analyis utilities
//----------*- C++ -*-===//
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
  /// cached. It expects to be passed a FuncOp operation where it can loop over
  /// its blocks (mimicking the BBs of the CFG)
  ControlDependenceAnalysis(Operation *operation) {
    // type-cast it into mlir::func::FuncOp
    mlir::func::FuncOp funcOp = dyn_cast<mlir::func::FuncOp>(operation);
    if (funcOp) {
      identifyAllControlDeps(funcOp);
    } else {
      // type-cast it into mlir::ModuleOp
      ModuleOp modOp = dyn_cast<ModuleOp>(operation);
      if (modOp) {
        auto funcOps = modOp.getOps<mlir::func::FuncOp>();
        // call those in a loop and create a big structure
        for (mlir::func::FuncOp funcOp : llvm::make_early_inc_range(funcOps)) {
          identifyAllControlDeps(funcOp);
        }
      } else
        // report an error indicating that the anaylsis is instantiated over
        // an inappropriate operation
        llvm::errs() << "ControlDependenceAnalysis is instantiated over an "
                        "operation that is not FuncOp or ModuleOp!\n";
    }
  };

  // All public functions are accessible from outside of the pass and require
  // the outside to specify the index of the funcOp of interest. If the outside
  // has a func::FuncOp rather than a ModuleOp, they should just pass a 0

  // return all BBs that the block in the argument, belonging to the funcOp of
  // the passed idx, is control dependent on
  void
  calculateBlockControlDeps(Block *block, int funcOp_idx,
                            SmallVector<Block *, 4> &returned_control_deps);

  // return only forward dependencies (i.e., excluding loop exits) that the
  // block in the argument, belonging to the funcOp of the passed idx, is
  // control dependent on
  void calculateBlockForwardControlDeps(
      Block *block, int funcOp_idx,
      SmallVector<Block *, 4> &returned_forward_control_deps);

  // this function allows for changing the addresses of the blocks
  void adjustBlockPtr(int funcOp_idx, Block *new_block);

  // loops over the blocks and prints to the terminal the block's control
  // dependencies of the passed function
  void printAllBlocksDeps(int funcOp_idx);
  void printForwardBlocksDeps(int funcOp_idx);

  /// Invalidation hook to keep the analysis cached across passes. Returns true
  /// if the analysis should be invalidated and fully reconstructed the next
  /// time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<ControlDependenceAnalysis>();
  }

private:
  // For every func::FuncOp, every basic block from the CFG is mapped to a
  // vector containing all basic blocks that it is control dependent on We are
  // using SmallVector because it is more efficient than std::vector which use
  // heap allocations If the content of SmallVec tor exceeds 4, it will still be
  // functional but will start using the heap
  SmallVector<DenseMap<Block *, SmallVector<Block *, 4>>, 4>
      all_control_deps_maps;

  // For every func::FuncOp, contains only the forward control dependencies,
  // excluding the loop exit conditions
  SmallVector<DenseMap<Block *, SmallVector<Block *, 4>>, 4>
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
      SmallVector<SmallVector<mlir::DominanceInfoNode *, 4>, 4>
          *traversed_nodes);

  // recursive function called inside enumeratePathsInPostDomTree
  void enumeratePathsInPostDomTreeUtil(
      mlir::DominanceInfoNode *start_node, mlir::DominanceInfoNode *end_node,
      DenseMap<mlir::DominanceInfoNode *, bool> is_visited,
      SmallVector<mlir::DominanceInfoNode *, 4> path, int path_index,
      SmallVector<SmallVector<mlir::DominanceInfoNode *, 4>, 4>
          *traversed_nodes);

  // adjusts the dependencies of each block to include nested dependencies
  // (i.e., the dependencies of its depenendencies)
  void
  addDepsOfDeps(mlir::func::FuncOp &funcOp,
                DenseMap<Block *, SmallVector<Block *, 4>> &control_deps_map);

  // helper function called inside adjustBlockPtr() function
  void compareNamesAndModifyBlockPtr(Block *new_block, Block *old_block,
                                     SmallVector<Block *, 4> old_block_deps);
};

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_CONTROLDEPENDENCEANALYSIS_H