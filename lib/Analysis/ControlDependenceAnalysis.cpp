//===- ControlDependenceAnalysis.h - Control dependence analyis *--- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements some utility functions useful to analyzing the control
// dependencies between basic blocks of the CFG. The control dependendies are
// calculated using the algorithm from the following paper
//   J.Ferrante, K.J. Ottenstein, and J. D. Warren, "The Program Dependence
//   Graph and its Use in Optimizations", ACM Trans. Program. Lang. Syst., vol.
//   9, pp. 319-349, 1987.
//
// According to def. 3 in the paper: "A node Y in a CFG is control dependant on
// another node Y iff (1) there exists a path P from X to Y with any Z different
// from X and Y post-dominated by Y and (2) X is not post-dominated by Y".
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::func;
using namespace dynamatic;

using PathInDomTree = SmallVector<DominanceInfoNode *>;
using PostDomTree = llvm::DominatorTreeBase<Block, true>;

/// Utility function to DFS inside the post-dominator tree and find the path
/// from a start node to a destination, if exists. Returns true in that case,
/// false otherwise
template <typename FunctionType>
static bool enumeratePathsInPostDomTree(DominanceInfoNode *startNode,
                                        DominanceInfoNode *endNode,
                                        PathInDomTree &currentPath) {
  currentPath.push_back(startNode);

  // If we are at the end of a path, then add it to the set of found paths
  if (startNode == endNode)
    return true;

  // For each of the successors of `startNode`, try each descendent until
  // `endNode` is found
  for (auto *iter = startNode->begin(); iter < startNode->end(); iter++) {
    if (enumeratePathsInPostDomTree<FunctionType>(*iter, endNode, currentPath))
      return true;
  }

  // Since at this point that was not the correct direction, pop the start node
  // and back trace
  currentPath.pop_back();
  return false;
}

/// Get the paths in the post dominator tree from a start node to and end node.
template <typename FunctionType>
static void
enumeratePathsInPostDomTree(Block *startBlock, Block *endBlock, Region *funcReg,
                            PostDomTree *postDomTree, PathInDomTree &path) {

  DominanceInfoNode *startNode = postDomTree->getNode(startBlock);
  DominanceInfoNode *endNode = postDomTree->getNode(endBlock);

  enumeratePathsInPostDomTree<FunctionType>(startNode, endNode, path);
}

template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::identifyAllControlDeps(
    FunctionType &funcOp) {

  // Get post-domination information
  Region &funcReg = funcOp.getRegion();
  PostDominanceInfo postDomInfo;
  PostDomTree &postDomTree = postDomInfo.getDomTree(&funcReg);

  // Consider each pair of successive block in the CFG
  for (Block &bb : funcReg.getBlocks()) {
    for (Block *successor : bb.getSuccessors()) {

      if (postDomInfo.properlyPostDominates(successor, &bb))
        continue;

      Block *leastCommonAnc =
          postDomInfo.findNearestCommonDominator(successor, &bb);

      // Loop case
      if (leastCommonAnc == &bb)
        blocksControlDeps[&bb].allControlDeps.insert(&bb);

      // In the post dominator tree, all the nodes from `leastCommonAnc` to
      // `successor` should be control dependent on `block`
      blocksControlDeps[successor].allControlDeps.insert(&bb);

      PathInDomTree pathFromLeastCommonAncToSuccessor;
      enumeratePathsInPostDomTree<FunctionType>(
          leastCommonAnc, successor, &funcReg, &postDomTree,
          pathFromLeastCommonAncToSuccessor);

      for (DominanceInfoNode *domInfo : pathFromLeastCommonAncToSuccessor) {
        Block *blockInPath = domInfo->getBlock();

        // Skip the nodes that we have already taken care of above
        if (blockInPath == leastCommonAnc || blockInPath == &bb ||
            blockInPath == successor)
          continue;

        blocksControlDeps[blockInPath].allControlDeps.insert(&bb);
      }
    }
  }

  // Include nested dependencies to the analysis
  addDepsOfDeps(funcOp);

  // Extract the forward dependencies out of all the control dependencies
  identifyForwardControlDeps(funcOp);
}

template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::addDepsOfDeps(
    FunctionType &funcOp) {

  // For each block, consider each of its dependencies (`oneDep`) and move each
  // of its dependencies into block's
  for (Block &block : funcOp.getBlocks()) {
    BlockControlDeps blockControlDeps = blocksControlDeps[&block];
    for (auto &oneDep : blockControlDeps.allControlDeps) {
      DenseSet<Block *> &oneDepDeps = blocksControlDeps[oneDep].allControlDeps;
      for (auto &oneDepDep : oneDepDeps)
        blocksControlDeps[&block].allControlDeps.insert(oneDepDep);
    }
  }
}

template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::identifyForwardControlDeps(
    FunctionType &funcOp) {
  Region &funcReg = funcOp.getRegion();

  // Get dominance, post-dominance and loop information
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
  llvm::DominatorTreeBase<Block, false> &domTree = domInfo.getDomTree(&funcReg);
  CFGLoopInfo li(domTree);

  for (Block &block : funcReg.getBlocks()) {

    // Consider all block's dependencies
    for (Block *oneDep : blocksControlDeps[&block].allControlDeps) {

      CFGLoop *loop = li.getLoopFor(oneDep);

      // It is a forward control dependency if:
      // - `oneDep` is not in a loop;
      // - `oneDep` is not a loop exit or post-dominates `block`;
      // - `oneDep` is not a latch block.
      if (!loop || ((!loop->isLoopLatch(oneDep)) &&
                    (!loop->isLoopExiting(oneDep) ||
                     postDomInfo.properlyPostDominates(oneDep, &block))))
        blocksControlDeps[&block].forwardControlDeps.insert(oneDep);
    }
  }
}

template <typename FunctionType>
std::optional<DenseSet<Block *>>
ControlDependenceAnalysis<FunctionType>::getBlockAllControlDeps(
    Block *block) const {
  if (!blocksControlDeps.contains(block))
    return std::nullopt;

  return blocksControlDeps.lookup(block).allControlDeps;
}

template <typename FunctionType>
std::optional<DenseSet<Block *>>
ControlDependenceAnalysis<FunctionType>::getBlockForwardControlDeps(
    Block *block) const {
  if (!blocksControlDeps.contains(block))
    return std::nullopt;

  return blocksControlDeps.lookup(block).forwardControlDeps;
}

template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::printAllBlocksDeps() const {

  DEBUG_WITH_TYPE(
      "CONTROL_DEPENDENCY_ANALYSIS",
      llvm::dbgs() << "\n*********************************\n\n";
      for (auto &elem : blocksControlDeps) {
        Block *block = elem.first;
        block->printAsOperand(llvm::dbgs());
        llvm::dbgs() << " is control dependent on: ";

        auto blockDeps = elem.second;

        for (auto &oneDep : blockDeps.allControlDeps) {
          oneDep->printAsOperand(llvm::dbgs());
          llvm::dbgs() << ", ";
        }

        llvm::dbgs() << "\n";
      } llvm::dbgs()
      << "\n*********************************\n";);
}

namespace dynamatic {

// Explicit template instantiation
template class ControlDependenceAnalysis<mlir::func::FuncOp>;
template class ControlDependenceAnalysis<handshake::FuncOp>;

} // namespace dynamatic
