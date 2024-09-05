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
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::func;
using namespace dynamatic;

using PathInDomTree = SmallVector<DominanceInfoNode *, 6>;

template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::identifyAllControlDeps(
    FunctionType &funcOp) {
  Region &funcReg = funcOp.getRegion();

  // Get information about post-dominance
  PostDominanceInfo postDomInfo;
  // Get the post-dominance tree
  llvm::DominatorTreeBase<Block, true> &postDomTree =
      postDomInfo.getDomTree(&funcReg);

  // Declare a map from the blocks of this function to their control
  // dependencies
  BlockControlDepsMap blockControlDepsMap;

  // Nested loop over the control flow edges connnecting the different blocks of
  // this region
  for (Block &block : funcReg.getBlocks()) {
    for (Block *blockSucc : block.getSuccessors()) {
      if (!postDomInfo.properlyPostDominates(blockSucc, &block)) {
        Block *leastCommonAnc =
            postDomInfo.findNearestCommonDominator(blockSucc, &block);

        if (leastCommonAnc == &block) {
          // Loop case
          blockControlDepsMap[&block].allControlDeps.insert(&block);
        }

        // All nodes in the post-dominator tree on the path from the
        // "least_common_anc" to "block_succ" (including "block_succ")
        // should be control dependent on "block"
        // Easy case of block_succ
        blockControlDepsMap[blockSucc].allControlDeps.insert(&block);

        // Enumerate all paths between "least_common_anc" and "block_succ"
        SmallVector<PathInDomTree, 6> allPathsFromLeastCommonAncToBlockSucc;
        enumeratePathsInPostDomTree(leastCommonAnc, blockSucc, &funcReg,
                                    &postDomTree,
                                    &allPathsFromLeastCommonAncToBlockSucc);

        for (const PathInDomTree &path :
             allPathsFromLeastCommonAncToBlockSucc) {
          for (DominanceInfoNode *domInfo : path) {
            Block *blockInPath = domInfo->getBlock();

            // Skip the nodes that we have already taken care of above
            if (blockInPath == leastCommonAnc || blockInPath == &block ||
                blockInPath == blockSucc)
              continue;

            blockControlDepsMap[blockInPath].allControlDeps.insert(&block);
          }
        }
      }
    }
  }

  // Up to this point, we have correct direct dependencies that do not
  // include nested dependencies (i.e., dependencies of a block's
  // dependencies) call the following function to include nested
  // dependencies
  addDepsOfDeps(funcOp, blockControlDepsMap);

  // We are done calculating the dependencies of all blocks of this funcOp so
  // store them in the map
  this->blocksControlDeps = blockControlDepsMap;

  // Extract the forward dependencies for this function
  identifyForwardControlDeps(funcOp);
}

template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::enumeratePathsInPostDomTree(
    Block *startBlock, Block *endBlock, Region *funcReg,
    llvm::DominatorTreeBase<Block, true> *postDomTree,
    SmallVector<PathInDomTree, 6> *traversedNodes) {

  DominanceInfoNode *startNode = postDomTree->getNode(startBlock);
  DominanceInfoNode *endNode = postDomTree->getNode(endBlock);

  DenseMap<DominanceInfoNode *, bool> isVisited;
  for (Block &block : funcReg->getBlocks())
    isVisited[postDomTree->getNode(&block)] = false;

  PathInDomTree path;
  path.resize(funcReg->getBlocks().size());
  unsigned pathIndex = 0;

  enumeratePathsInPostDomTreeUtil(startNode, endNode, isVisited, path,
                                  pathIndex, traversedNodes);
}

// DFS to return all nodes in the path between the start_node and end_node
// (not including start_node and end_node) in the postDom tree
template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::enumeratePathsInPostDomTreeUtil(
    DominanceInfoNode *startNode, DominanceInfoNode *endNode,
    DenseMap<DominanceInfoNode *, bool> isVisited, PathInDomTree path,
    int pathIndex, SmallVector<PathInDomTree, 6> *traversedNodes) {
  isVisited[startNode] = true;
  path[pathIndex] = startNode;
  pathIndex++;

  // if start is same as end, we have completed one path so push it to
  // traversed_nodes
  if (startNode == endNode) {
    // slice of the path from its beginning until the path_index
    PathInDomTree actualPath;
    for (auto i = 0; i < pathIndex; i++)
      actualPath.push_back(path[i]);
    traversedNodes->push_back(actualPath);

  } else {
    // loop over the children of start_node
    for (DominanceInfoNode::iterator iter = startNode->begin();
         iter < startNode->end(); iter++) {
      if (!isVisited[*iter]) {
        enumeratePathsInPostDomTreeUtil(*iter, endNode, isVisited, path,
                                        pathIndex, traversedNodes);
      }
    }
  }

  // remove this node from path and mark it as unvisited
  pathIndex--;
  isVisited[startNode] = false;
}

template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::addDepsOfDeps(
    FunctionType &funcOp, BlockControlDepsMap &blockControlDepsMap) {
  Region &funcReg = funcOp.getRegion();

  for (Block &block : funcReg.getBlocks()) {
    BlockControlDeps blockControlDeps = blockControlDepsMap[&block];
    // loop on the dependencies of one block
    for (auto &oneDep : blockControlDeps.allControlDeps) {
      DenseSet<Block *> &oneDepDeps =
          blockControlDepsMap[oneDep].allControlDeps;
      // loop on the dependencies of one_dep
      for (auto &oneDepDep : oneDepDeps)
        blockControlDepsMap[&block].allControlDeps.insert(oneDepDep);
    }
  }
}

template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::identifyForwardControlDeps(
    FunctionType &funcOp) {
  Region &funcReg = funcOp.getRegion();
  DominanceInfo domInfo;
  llvm::DominatorTreeBase<Block, false> &domTree = domInfo.getDomTree(&funcReg);
  // Get loop information to eliminate loop exits from the dependencies
  CFGLoopInfo li(domTree);

  // Get information about post-dominance
  PostDominanceInfo postDomInfo;

  for (Block &block : funcReg.getBlocks()) {
    // Extract the dependencies of this block to adjust them by removing
    // loop exit conditions, if any
    DenseSet<Block *> &allControlDeps =
        blocksControlDeps[&block].allControlDeps;

    for (Block *oneDep : allControlDeps) {
      CFGLoop *loop = li.getLoopFor(oneDep);
      if (!loop) {
        // indicating that the one_dep is not inside any loop, so it must be
        // a forward dependency
        blocksControlDeps[&block].forwardControlDeps.insert(oneDep);
      } else {
        // indicating that the one_dep is inside a loop,
        // to decide if it is a forward dep or not, compare it against all
        // of the exits and latches of this loop
        bool notForward = false;
        // check if one_dep is an exit of the loop
        SmallVector<Block *> loopExitBlocks;
        loop->getExitingBlocks(loopExitBlocks);
        for (Block *loopExit : loopExitBlocks) {
          if (loopExit == oneDep &&
              !postDomInfo.properlyPostDominates(loopExit, &block)) {
            notForward = true;
            // it is not a forward dependency so no need to contiue
            // looping
            break;
          }
        }
        // check if one_dep is a latch of the loop
        SmallVector<Block *> loopLatchBlocks;
        loop->getLoopLatches(loopLatchBlocks);
        for (auto &loopLatch : loopLatchBlocks) {
          if (loopLatch == oneDep) {
            notForward = true;
            // it is not a forward dependency so no need to contiue
            // looping
            break;
          }
        }

        if (!notForward)
          blocksControlDeps[&block].forwardControlDeps.insert(oneDep);
      }
    }
  }
}

template <typename FunctionType>
LogicalResult ControlDependenceAnalysis<FunctionType>::getBlockAllControlDeps(
    Block *block, DenseSet<Block *> &allControlDeps) const {
  if (!blocksControlDeps.contains(block)) {
    llvm::errs() << "call to ControlDependenceAnalysis::getBlockAllControlDeps "
                    "on a block which is in not in the associated funcOp";
    return failure();
  }
  allControlDeps = blocksControlDeps.lookup(block).allControlDeps;
  return success();
}

template <typename FunctionType>
LogicalResult
ControlDependenceAnalysis<FunctionType>::getBlockForwardControlDeps(
    Block *block, DenseSet<Block *> &forwardControlDeps) const {
  if (!blocksControlDeps.contains(block)) {
    llvm::errs()
        << "call to ControlDependenceAnalysis::getBlockForwardControlDeps "
           "on a block which is in not in the associated funcOp";
    return failure();
  }
  forwardControlDeps = blocksControlDeps.lookup(block).forwardControlDeps;
  return success();
}

template <typename FunctionType>
void ControlDependenceAnalysis<FunctionType>::printAllBlocksDeps() const {

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
  }
  llvm::dbgs() << "\n*********************************\n";
}

namespace dynamatic {

// Explicit template instantiation
template class ControlDependenceAnalysis<mlir::func::FuncOp>;
template class ControlDependenceAnalysis<handshake::FuncOp>;

} // namespace dynamatic
