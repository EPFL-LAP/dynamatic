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
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::func;
using namespace dynamatic;

using PathInDomTree = SmallVector<DominanceInfoNode *, 6>;

void ControlDependenceAnalysis::identifyAllControlDeps(FuncOp &funcOp) {
  Region &funcReg = funcOp.getRegion();
  // functions made up of 1 basic block do not have any dependencies
  if (funcReg.hasOneBlock())
    return;

  // Get information about post-dominance
  PostDominanceInfo postDomInfo;
  // Get the post-dominance tree
  llvm::DominatorTreeBase<Block, true> &postDomTree =
      postDomInfo.getDomTree(&funcReg);

  // Declare a map from the blocks of this function to their control
  // dependencies
  BlockControlDepsMap block_control_deps_map;

  // Nested loop over the control flow edges connnecting the different blocks of
  // this region
  for (Block &block : funcReg.getBlocks()) {
    for (Block *block_succ : block.getSuccessors()) {
      if (!postDomInfo.properlyPostDominates(block_succ, &block)) {
        Block *least_common_anc =
            postDomInfo.findNearestCommonDominator(block_succ, &block);

        if (least_common_anc == &block) {
          // Loop case
          block_control_deps_map[&block].all_control_deps.insert(&block);
        }

        // All nodes in the post-dominator tree on the path from the
        // "least_common_anc" to "block_succ" (including "block_succ")
        // should be control dependent on "block"
        // Easy case of block_succ
        block_control_deps_map[block_succ].all_control_deps.insert(&block);

        // Enumerate all paths between "least_common_anc" and "block_succ"
        SmallVector<PathInDomTree, 6> allPathsFromLeastCommonAncToBlockSucc;
        enumeratePathsInPostDomTree(least_common_anc, block_succ, &funcReg,
                                    &postDomTree,
                                    &allPathsFromLeastCommonAncToBlockSucc);

        for (PathInDomTree path : allPathsFromLeastCommonAncToBlockSucc) {
          for (DominanceInfoNode *domInfo : path) {
            Block *blockInPath = domInfo->getBlock();

            // Skip the nodes that we have already taken care of above
            if (blockInPath == least_common_anc || blockInPath == &block ||
                blockInPath == block_succ)
              continue;

            block_control_deps_map[blockInPath].all_control_deps.insert(&block);
          }
        }
      }
    }
  }

  // Up to this point, we have correct direct dependencies that do not
  // include nested dependencies (i.e., dependencies of a block's
  // dependencies) call the following function to include nested
  // dependencies
  addDepsOfDeps(funcOp, block_control_deps_map);

  // We are done calculating the dependencies of all blocks of this funcOp so
  // store them in the map
  func_blocks_control_deps[funcOp] = block_control_deps_map;

  // Extract the forward dependencies for this function
  identifyForwardControlDeps(funcOp);
}

void ControlDependenceAnalysis::enumeratePathsInPostDomTree(
    Block *start_block, Block *end_block, Region *funcReg,
    llvm::DominatorTreeBase<Block, true> *postDomTree,
    SmallVector<PathInDomTree, 6> *traversed_nodes) {

  DominanceInfoNode *start_node = postDomTree->getNode(start_block);
  DominanceInfoNode *end_node = postDomTree->getNode(end_block);

  DenseMap<DominanceInfoNode *, bool> is_visited;
  for (Block &block : funcReg->getBlocks())
    is_visited[postDomTree->getNode(&block)] = false;

  PathInDomTree path;
  path.resize(funcReg->getBlocks().size());
  unsigned path_index = 0;

  enumeratePathsInPostDomTreeUtil(start_node, end_node, is_visited, path,
                                  path_index, traversed_nodes);
}

// DFS to return all nodes in the path between the start_node and end_node
// (not including start_node and end_node) in the postDom tree
void ControlDependenceAnalysis::enumeratePathsInPostDomTreeUtil(
    DominanceInfoNode *start_node, DominanceInfoNode *end_node,
    DenseMap<DominanceInfoNode *, bool> is_visited, PathInDomTree path,
    int path_index, SmallVector<PathInDomTree, 6> *traversed_nodes) {
  is_visited[start_node] = true;
  path[path_index] = start_node;
  path_index++;

  // if start is same as end, we have completed one path so push it to
  // traversed_nodes
  if (start_node == end_node) {
    // slice of the path from its beginning until the path_index
    PathInDomTree actual_path;
    for (auto i = 0; i < path_index; i++)
      actual_path.push_back(path[i]);
    traversed_nodes->push_back(actual_path);

  } else {
    // loop over the children of start_node
    for (DominanceInfoNode::iterator iter = start_node->begin();
         iter < start_node->end(); iter++) {
      if (!is_visited[*iter]) {
        enumeratePathsInPostDomTreeUtil(*iter, end_node, is_visited, path,
                                        path_index, traversed_nodes);
      }
    }
  }

  // remove this node from path and mark it as unvisited
  path_index--;
  is_visited[start_node] = false;
}

void ControlDependenceAnalysis::addDepsOfDeps(
    FuncOp &funcOp, BlockControlDepsMap &block_control_deps_map) {
  Region &funcReg = funcOp.getRegion();

  for (Block &block : funcReg.getBlocks()) {
    BlockControlDeps block_control_deps = block_control_deps_map[&block];
    // loop on the dependencies of one block
    for (auto &one_dep : block_control_deps.all_control_deps) {
      DenseSet<Block *> &one_dep_deps =
          block_control_deps_map[one_dep].all_control_deps;
      // loop on the dependencies of one_dep
      for (auto &one_dep_dep : one_dep_deps)
        block_control_deps_map[&block].all_control_deps.insert(one_dep_dep);
    }
  }
}

void ControlDependenceAnalysis::identifyForwardControlDeps(FuncOp &funcOp) {
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
    DenseSet<Block *> &all_control_deps =
        func_blocks_control_deps[funcOp][&block].all_control_deps;

    for (Block *one_dep : all_control_deps) {
      CFGLoop *loop = li.getLoopFor(one_dep);
      if (!loop) {
        // indicating that the one_dep is not inside any loop, so it must be
        // a forward dependency
        func_blocks_control_deps[funcOp][&block].forward_control_deps.insert(
            one_dep);
      } else {
        // indicating that the one_dep is inside a loop,
        // to decide if it is a forward dep or not, compare it against all
        // of the exits and latches of this loop
        bool not_forward = false;
        // check if one_dep is an exit of the loop
        SmallVector<Block *> loop_exitBlocks;
        loop->getExitingBlocks(loop_exitBlocks);
        for (Block *loop_exit : loop_exitBlocks) {
          if (loop_exit == one_dep &&
              !postDomInfo.properlyPostDominates(loop_exit, &block)) {
            not_forward = true;
            // it is not a forward dependency so no need to contiue
            // looping
            break;
          }
        }
        // check if one_dep is a latch of the loop
        SmallVector<Block *> loop_latchBlocks;
        loop->getLoopLatches(loop_latchBlocks);
        for (auto &loop_latch : loop_latchBlocks) {
          if (loop_latch == one_dep) {
            not_forward = true;
            // it is not a forward dependency so no need to contiue
            // looping
            break;
          }
        }

        if (!not_forward)
          func_blocks_control_deps[funcOp][&block].forward_control_deps.insert(
              one_dep);
      }
    }
  }
}

void ControlDependenceAnalysis::getBlockAllControlDeps(
    Block *block, FuncOp &funcOp, DenseSet<Block *> &all_control_deps) {
  all_control_deps = func_blocks_control_deps[funcOp][block].all_control_deps;
}

void ControlDependenceAnalysis::getBlockForwardControlDeps(
    Block *block, FuncOp &funcOp, DenseSet<Block *> &forward_control_deps) {
  forward_control_deps =
      func_blocks_control_deps[funcOp][block].forward_control_deps;
}
