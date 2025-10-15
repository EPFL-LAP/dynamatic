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

  DenseMap<Block *, unsigned> dfsNum;
  unsigned counter = 0;

  auto dfs = [&](auto &&self, Block *b) -> void {
    if (dfsNum.count(b))
      return;
    dfsNum[b] = counter++;
    for (Block *succ : b->getSuccessors())
      self(self, succ);
  };

  if (region.empty())
    return;

  dfs(dfs, &region.front());

  auto comesBeforeInCFG = [&](Block *a, Block *b) -> bool {
    return dfsNum[a] < dfsNum[b];
  };

  // oneDep is considered a forwardControlDep if it comes before block in the
  // CFG
  for (Block &block : region.getBlocks())
    for (Block *oneDep : blocksControlDeps[&block].allControlDeps)
      if (comesBeforeInCFG(oneDep, &block))
        blocksControlDeps[&block].forwardControlDeps.insert(oneDep);
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

DenseSet<Block *> dynamatic::getLocalConsDependence(Block *prod, Block *cons) {
  if (!prod || !cons)
    return DenseSet<Block *>{};
  Region *reg = prod->getParent();
  if (!reg || reg != cons->getParent())
    return DenseSet<Block *>{};

  // 1) Build the local subgraph G' with a single sink node Sk
  struct NodeInfo {
    SmallVector<Block *, 4> succs; // Successors in G' (excluding Sk)
    bool toSink = false;           // Whether this node has an edge to Sk
  };

  DenseMap<Block *, NodeInfo> G; // Only contains blocks in the local subgraph
  Block *SINK = nullptr; // Use nullptr to represent Sk (the unique exit)

  enum class Mark : int { kUnseen = 0, kSeen, kDone };
  DenseMap<Block *, Mark> mark;

  // Construction rules:
  //  1) Keep all edges u -> cons. The block `cons` itself is not expanded,
  //     but must always have an outgoing edge cons -> Sk.
  //  2) For any successor s == prod:
  //       - If prod == cons, connect u -> cons (and cons will later connect to
  //       Sk).
  //       - If prod != cons, connect u -> Sk (this represents the second visit
  //       to prod).
  //  3) For any block with no successors or successors outside the current
  //  region,
  //     connect that block -> Sk.
  //  4) If prod == cons and b == cons at the starting point (isStart == true),
  //     do NOT stop; expand successors once normally.
  std::function<void(Block *, bool)> build = [&](Block *b, bool isStart) {
    (void)G[b]; // Ensure the node exists in the map

    // Case: current block is `cons`
    // Always add an edge cons -> Sk
    if (b == cons) {
      G[b].toSink = true; // cons -> Sk always exists
      // At the starting node (prod == cons, first visit): keep exploring
      if (!(isStart && prod == cons)) {
        // For non-start or when prod != cons, stop expanding at cons
        mark[b] = Mark::kDone;
        return;
      }
      // Otherwise continue expanding successors normally
    }

    if (mark[b] == Mark::kDone)
      return;
    if (mark[b] == Mark::kSeen && !isStart)
      return;
    mark[b] = Mark::kSeen;

    auto succRange = b->getSuccessors();
    if (succRange.empty()) {
      // True end block: connect to Sk
      G[b].toSink = true;
      mark[b] = Mark::kDone;
      return;
    }

    for (Block *s : succRange) {
      // Out-of-region successors are treated as -> Sk
      if (!s || s->getParent() != reg) {
        G[b].toSink = true;
        continue;
      }

      // Successor is `prod`
      if (s == prod) {
        if (prod == cons) {
          // Second visit to prod is treated as reaching cons.
          // First connect to cons, then cons -> Sk will terminate the path.
          (void)G[cons];
          G[b].succs.push_back(cons);
        } else {
          // prod != cons: second visit to prod terminates directly -> Sk
          G[b].toSink = true;
        }
        continue;
      }

      // Successor is `cons`: keep the edge but do not expand cons
      if (s == cons) {
        (void)G[cons];
        G[b].succs.push_back(cons);
        continue; // Do not recursively expand cons
      }

      // Normal edge b -> s
      G[b].succs.push_back(s);
      build(s, /*isStart=*/false);
    }

    mark[b] = Mark::kDone;
  };

  build(prod, /*isStart=*/true);

  // If cons is unreachable in the local subgraph, return an empty set
  if (!G.contains(cons))
    return DenseSet<Block *>{};

  // Expand all "toSink" edges into explicit -> Sk edges
  SmallVector<Block *, 32> nodes;
  nodes.reserve(G.size() + 1);
  for (auto &kv : G)
    nodes.push_back(kv.first);
  nodes.push_back(SINK);

  DenseMap<Block *, SmallVector<Block *, 4>> succP, predP;
  for (Block *n : nodes) {
    succP[n];
    predP[n];
  }
  for (auto &kv : G) {
    Block *u = kv.first;
    for (Block *v : kv.second.succs) {
      succP[u].push_back(v);
      predP[v].push_back(u);
    }
    if (kv.second.toSink) {
      succP[u].push_back(SINK);
      predP[SINK].push_back(u);
    }
  }

  // 2) Compute post-dominance sets with respect to Sk
  DenseMap<Block *, DenseSet<Block *>> postdom;
  for (Block *n : nodes) {
    if (n == SINK) {
      postdom[n].insert(SINK);
    } else {
      for (Block *m : nodes)
        postdom[n].insert(m);
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Block *n : nodes) {
      if (n == SINK)
        continue;

      // inter = intersection of postdom(s) for all successors s of n
      DenseSet<Block *> inter;
      bool first = true;
      for (Block *s : succP[n]) {
        if (first) {
          inter = postdom[s];
          first = false;
        } else {
          DenseSet<Block *> tmp;
          for (Block *x : inter)
            if (postdom[s].contains(x))
              tmp.insert(x);
          inter.swap(tmp);
        }
      }

      DenseSet<Block *> newSet = inter;
      newSet.insert(n);

      // Check for changes
      if (newSet.size() != postdom[n].size()) {
        postdom[n].swap(newSet);
        changed = true;
        continue;
      }
      for (Block *x : newSet) {
        if (!postdom[n].contains(x)) {
          postdom[n].swap(newSet);
          changed = true;
          break;
        }
      }
    }
  }

  auto postDominates = [&](Block *x, Block *y) -> bool {
    auto it = postdom.find(y);
    return it != postdom.end() && it->second.contains(x);
  };

  // 3) Query CDG direct predecessors (without explicitly building CDG)
  // Ferrante–Ottenstein–Warren criterion:
  //   ∃ Y->Z such that X ∈ postdom(Z) and X ∉ postdom(Y)
  //   -> add edge Y -> X in CDG
  auto predsInCDG = [&](Block *X, DenseSet<Block *> &out) {
    for (Block *Y : nodes) {
      if (Y == SINK)
        continue;
      if (postDominates(X, Y))
        continue;
      for (Block *Z : succP[Y]) {
        if (postDominates(X, Z)) {
          out.insert(Y);
          break;
        }
      }
    }
  };

  // 4) Reverse closure: collect all control ancestors of `cons`
  DenseSet<Block *> upstream, frontier;
  SmallVector<Block *, 16> stack;

  // Direct control predecessors of cons
  predsInCDG(cons, frontier);
  for (Block *p : frontier) {
    upstream.insert(p);
    stack.push_back(p);
  }

  // BFS/DFS over CDG to get transitive closure (all upstream control points)
  while (!stack.empty()) {
    Block *x = stack.back();
    stack.pop_back();
    DenseSet<Block *> parents;
    predsInCDG(x, parents);
    for (Block *p : parents) {
      if (!upstream.contains(p)) {
        upstream.insert(p);
        stack.push_back(p);
      }
    }
  }

  return upstream;
}

void dynamatic::ControlDependenceAnalysis::printAllBlocksDeps() const {

  DEBUG_WITH_TYPE(
      "CONTROL_DEPENDENCY_ANALYSIS",
      llvm::dbgs() << "\n*********************************\n\n";
      for (auto &elem
           : blocksControlDeps) {
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
