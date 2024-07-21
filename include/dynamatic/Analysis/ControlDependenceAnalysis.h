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

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/AnalysisManager.h"

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
  ControlDependenceAnalysis(mlir::Operation *operation) {
    // type-cast it into mlir::func::FuncOp
    mlir::func::FuncOp funcOp = mlir::dyn_cast<mlir::func::FuncOp>(operation);
    if (funcOp) {
      identifyAllControlDeps(funcOp);
      identifyForwardControlDeps(funcOp);
    } else {
      // report an error indicating that the anaylsis is instantiated over an
      // inappropriate operation
      llvm::errs() << "ControlDependenceAnalysis is instantiated over an "
                      "operation that is not FuncOp!\n";
    }
  };

  // this function allows for changing the addresses of the blocks  
  void adjustBlocksPtrs(Region &funcReg); 

  // return all BBs that the block in the argument is control dependent on
  void returnAllControlDeps(
      mlir::Block *block,
      llvm::SmallVector<mlir::Block *, 4> &returned_all_control_deps);

  // return only forward dependencies (i.e., excluding loop exits) that the
  // block in the argument is control dependent on
  void returnForwardControlDeps(
      mlir::Block *block,
      llvm::SmallVector<mlir::Block *, 4> &returned_forward_control_deps);

  // loops over the blocks and prints to the terminal the block's control
  // dependencies
  void printAllBlocksDeps(mlir::func::FuncOp &funcOp);

  void printForwardBlocksDeps(mlir::func::FuncOp &funcOp);

  /// Invalidation hook to keep the analysis cached across passes. Returns true
  /// if the analysis should be invalidated and fully reconstructed the next
  /// time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<ControlDependenceAnalysis>();
  }

private:
  // Every basic block from the CFG is mapped to a vector containing all basic
  // blocks that it is control dependent on We are using SmallVector because it
  // is more efficient than std::vector which use heap allocations If the
  // content of SmallVec tor exceeds 4, it will still be functional but will
  // start using the heap
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 4>>
      all_control_deps_map;

  // contains only the forward control dependencies, excluding the loop exit
  // conditions
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 4>>
      forward_control_deps_map;

  // Simply fill the all_control_deps_map
  void identifyAllControlDeps(
      mlir::func::FuncOp &funcOp); // called inside the constructor

  // Simply extract the forward_control_deps_map from the all_control_deps_map
  void identifyForwardControlDeps(
      mlir::func::FuncOp &funcOp); // called inside the constructor

  // Returns all postDominator tree nodes between start_node and end_node in the
  // postDominator tree
  void traversePostDomTree(
      mlir::Block *start_block, mlir::Block *end_block, mlir::Region *funcReg,
      llvm::DominatorTreeBase<mlir::Block, true> *postDomTree,
      llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode *, 4>, 4>
          *traversed_nodes);

  // adjusts the dependencies of each block to include nested dependencies
  // (i.e., the dependencies of its depenendencies)
  void addDepsOfDeps(mlir::func::FuncOp &funcOp);

  // recursive function called inside traversePostDomTree
  void traversePostDomTreeUtil(
      mlir::DominanceInfoNode *start_node, mlir::DominanceInfoNode *end_node,
      llvm::DenseMap<mlir::DominanceInfoNode *, bool> is_visited,
      llvm::SmallVector<mlir::DominanceInfoNode *, 4> path, int path_index,
      llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode *, 4>, 4>
          *traversed_nodes);
};

// takes any traversed_nodes structure and prints it to the terminal
void printPostDomTreeTraversal(
    llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode *, 4>, 4>
        traversed_nodes);

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_CONTROLDEPENDENCEANALYSIS_H