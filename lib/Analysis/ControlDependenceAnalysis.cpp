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
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace dynamatic;
using namespace mlir;
#define DEBUG_TYPE "control-dependence-analysis"

using PathInDomTree = SmallVector<DominanceInfoNode *>;
using PostDomTree = llvm::DominatorTreeBase<Block, true>;

ControlDependenceAnalysis::ControlDependenceAnalysis(Region &region) {
  identifyAllControlDeps(region);
}

ControlDependenceAnalysis::ControlDependenceAnalysis(Operation *operation) {

  // Only one function should be present in the module, excluding external
  // functions
  unsigned functionsCovered = 0;

  // The analysis can be instantiated either over a module containing one
  // function only or over a function
  if (ModuleOp modOp = dyn_cast<ModuleOp>(operation); modOp) {
    for (func::FuncOp funcOp : modOp.getOps<func::FuncOp>()) {

      // Skip if external
      if (funcOp.isExternal())
        continue;

      // Analyze the function
      if (!functionsCovered) {
        identifyAllControlDeps(funcOp.getRegion());
        functionsCovered++;
      } else {
        llvm::errs() << "[CDA] Too many functions to handle in the module";
      }
    }
  } else if (func::FuncOp fOp = dyn_cast<func::FuncOp>(operation); fOp) {
    identifyAllControlDeps(fOp.getRegion());
    functionsCovered = 1;
  }

  // report an error indicating that the analysis is instantiated over
  // an inappropriate operation
  if (functionsCovered != 1)
    llvm::errs() << "[CDA] Control Dependency Analysis failed due to a wrong "
                    "input type\n";
};

/// Utility function to DFS inside the post-dominator tree and find the path
/// from a start node to a destination, if exists. Returns true in that case,
/// false otherwise
static bool enumeratePathsInPostDomTree(DominanceInfoNode *startNode,
                                        DominanceInfoNode *endNode,
                                        PathInDomTree &currentPath) {
  currentPath.push_back(startNode);

  // If we are at the end of a path, then add it to the set of found paths
  if (startNode == endNode)
    return true;

  // For each of the successors of `startNode`, try each descendent until
  // `endNode` is found
  for (auto *node : startNode->children()) {
    if (enumeratePathsInPostDomTree(node, endNode, currentPath))
      return true;
  }

  // Since at this point that was not the correct direction, pop the start node
  // and back trace
  currentPath.pop_back();
  return false;
}

/// Get the paths in the post dominator tree from a start node to and end node.
static void enumeratePathsInPostDomTree(Block *startBlock, Block *endBlock,
                                        Region *funcReg,
                                        PostDomTree *postDomTree,
                                        PathInDomTree &path) {

  DominanceInfoNode *startNode = postDomTree->getNode(startBlock);
  DominanceInfoNode *endNode = postDomTree->getNode(endBlock);

  enumeratePathsInPostDomTree(startNode, endNode, path);
}

void dynamatic::ControlDependenceAnalysis::identifyAllControlDeps(
    Region &region) {

  if (region.getBlocks().size() == 1)
    return;

  // Get post-domination information
  PostDominanceInfo postDomInfo;
  PostDomTree &postDomTree = postDomInfo.getDomTree(&region);

  // Consider each pair of successive block in the CFG
  for (Block &bb : region.getBlocks()) {
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
      enumeratePathsInPostDomTree(leastCommonAnc, successor, &region,
                                  &postDomTree,
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
  addDepsOfDeps(region);

  // Extract the forward dependencies out of all the control dependencies
  identifyForwardControlDeps(region);
}

void dynamatic::ControlDependenceAnalysis::addDepsOfDeps(Region &region) {

  // For each block, consider each of its dependencies (`oneDep`) and move each
  // of its dependencies into block's
  for (Block &block : region.getBlocks()) {
    BlockControlDeps blockControlDeps = blocksControlDeps[&block];
    for (auto &oneDep : blockControlDeps.allControlDeps) {
      DenseSet<Block *> &oneDepDeps = blocksControlDeps[oneDep].allControlDeps;
      for (auto &oneDepDep : oneDepDeps)
        blocksControlDeps[&block].allControlDeps.insert(oneDepDep);
    }
  }
}

void dynamatic::ControlDependenceAnalysis::identifyForwardControlDeps(
    Region &region) {

  // Get dominance, post-dominance and loop information
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
  llvm::DominatorTreeBase<Block, false> &domTree = domInfo.getDomTree(&region);
  CFGLoopInfo li(domTree);

  for (Block &block : region.getBlocks()) {

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

std::optional<DenseSet<Block *>>
dynamatic::ControlDependenceAnalysis::getBlockAllControlDeps(
    Block *block) const {
  if (!blocksControlDeps.contains(block))
    return std::nullopt;

  return blocksControlDeps.lookup(block).allControlDeps;
}

std::optional<DenseSet<Block *>>
dynamatic::ControlDependenceAnalysis::getBlockForwardControlDeps(
    Block *block) const {
  if (!blocksControlDeps.contains(block))
    return std::nullopt;

  return blocksControlDeps.lookup(block).forwardControlDeps;
}

// Return the map of the control dependencies as stored in the class
ControlDependenceAnalysis::BlockControlDepsMap
dynamatic::ControlDependenceAnalysis::getAllBlockDeps() const {
  return blocksControlDeps;
}

void dynamatic::ControlDependenceAnalysis::printAllBlocksDeps() const {

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
