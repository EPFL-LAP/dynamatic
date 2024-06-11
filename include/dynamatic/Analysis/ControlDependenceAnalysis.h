//===- ControlDependenceAnalysis.h - Control dependence analyis utilities ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions useful to analyzing the control dependencies between basic blocks of the CFG.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_ANALYSIS_NAMEANALYSIS
#define DYNAMATIC_ANALYSIS_NAMEANALYSIS

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/IR/Dominance.h"


namespace dynamatic {

/// Analysis to maintain the control dependencies of basic blocks throughout the IR's lifetime.
/// Query at the beginning of Dynamatic passes using
/// `getAnalysis<ControlDependenceAnalysis>()` and cache it to avoid recomputations for
/// further passes using `markAnalysesPreserved<ControlDependenceAnalysis>()` at the end.
///
class ControlDependenceAnalysis {
public:
  /// Constructor called automatically by `getAnalysis<ControlDependenceAnalysis>()` if the
  /// analysis is not already cached. 
  /// It expects to be passed a FuncOp operation where it can loop over its blocks (mimicking the BBs of the CFG)
  ControlDependenceAnalysis(mlir::Operation*operation){//(mlir::func::FuncOp &funcOp) {
    identifyAllControlDeps(operation);
  };

  // return all BBs that the argument block is control dependent on
  void returnControlDeps(mlir::Block* block, llvm::SmallVector<mlir::Block*, 4>& returned_control_deps);

  /// Invalidation hook to keep the analysis cached across passes. Returns true
  /// if the analysis should be invalidated and fully reconstructed the next
  /// time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<ControlDependenceAnalysis>();
  }

private:
  // Every basic block from the CFG is mapped to a vector containing all basic blocks that it is control dependent on
  // We are using SmallVector because it is more efficient than std::vector which use heap allocations
  // If the content of SmallVec tor exceeds 4, it will still be functional but will start using the heap
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block*, 4>> control_deps_map;

  // Simply fill the control_deps_map 
  void identifyAllControlDeps(mlir::Operation*operation);

  // recursive function called inside traversePostDomTree
  void traversePostDomTreeUtil(mlir::DominanceInfoNode *start_node, mlir::DominanceInfoNode *end_node, llvm::DenseMap<mlir::DominanceInfoNode*, bool> is_visited, llvm::SmallVector<mlir::DominanceInfoNode*, 4> path, int path_index, llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode*, 4> , 4>*traversed_nodes);

  // Returns all postDominator tree nodes between start_node and end_node in the postDominator tree
  void traversePostDomTree(mlir::Block *start_block, mlir::Block *end_block, mlir::Region *funcReg, llvm::DominatorTreeBase<mlir::Block, true> *postDomTree, llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode*, 4> , 4>*traversed_nodes);
};

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_CDGANALYSIS