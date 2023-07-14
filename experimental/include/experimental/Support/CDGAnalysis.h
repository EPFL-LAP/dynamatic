//===- CDGAnalysis.h - Exp. support for CDG analysis -------*- C++ -*-===//
//
// This file contains the function for CDG analysis.
//
//===-----------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H
#define EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H

#include <set>

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

// Preorder post-dominance tree traversal
void PostDomTreeTraversal(DominanceInfoNode *node, unsigned level) {
  if (!node || level < 0) return;

  // visit
  DominanceInfoNode* iPostDomInf = node->getIDom(); // parent(node) in post dom tree

  // Exit node (the only node on level 0) does not have immediate post-dominator.
  // Immediate post-dominator of nodes on level 1 is the exit node.
  if (level > 1 && iPostDomInf) {
    Block* iPostDomBlock = iPostDomInf->getBlock();
    iPostDomBlock->print(llvm::outs());
  }
  
  // end of visit

  for (typename DominanceInfoNode::iterator i = node->begin(), end = node->end();
       i != end; ++i) {
    PostDomTreeTraversal(*i, level + 1);
  }
}

struct Edge {
  DominanceInfoNode* from; // A
  DominanceInfoNode* to;   // B

  Edge(DominanceInfoNode* from, DominanceInfoNode* to) {
    this->from = from;
    this->to = to;
  }

  DominanceInfoNode* findLowestCommonAncestor() {
    std::set<DominanceInfoNode*> ancestorsA;
    unsigned level;

    // Traverse upwards from node A and store its ancestors in the set.
    level = from->getLevel();
    for (DominanceInfoNode* curr = from; level > 0; --level, curr = curr->getIDom()) {
      ancestorsA.insert(curr);
    }
    
    // Traverse upwards from node B until a common ancestor is found.
    level = to->getLevel();
    for (DominanceInfoNode* curr = to; level > 0; --level, curr = curr->getIDom()) {
      if (ancestorsA.find(curr) != ancestorsA.end()) {
        // Lowest common ancestor
        return curr;
      }
    }

    return nullptr;
  }
};

// Traversal of the CFG that creates set of graph edges (A,B) so A is NOT
// post-dominated by B.
void CFGTraversal(Block* rootBlock, std::set<Block*>* visitedSet, std::set<Edge*>* edgeSet, 
                  PostDominanceInfo* postDomInfo, llvm::DominatorTreeBase<Block, true>& postDomTree) {
  if (!rootBlock) {
    return;
  }
  Block* curr = rootBlock;

  // Mark current node as visited.
  visitedSet->insert(curr);

  // end visit

  for(Block* successor : curr->getSuccessors()) {
    // Check if successor is visited.
    if (visitedSet->find(successor) != visitedSet->end()) {
      // Node is already visited.
      continue;
    }

    // Check if curr is not post-dominated by his successor.
    if (postDomInfo->properlyPostDominates(successor, curr)) {
      continue;
    }

    // Form the edge struct and add it to the set.
    DominanceInfoNode* succPostDomNode = postDomTree.getNode(successor);
    DominanceInfoNode* currPostDomNode = postDomTree.getNode(curr);

    Edge* edge = new Edge(currPostDomNode, succPostDomNode);
    edgeSet->insert(edge);

    curr->print(llvm::outs());
    successor->print(llvm::outs());
    llvm::outs() << "Edge added to the set.\n";

    CFGTraversal(successor, visitedSet, edgeSet, postDomInfo, postDomTree);
  }
}

namespace dynamatic {
namespace experimental {

  void hello();

  // CDG analysis function
  LogicalResult CDGAnalysis(func::FuncOp funcOp, MLIRContext* ctx) {

    Region& funcReg = funcOp.getRegion();
    Block& rootBlockCFG = funcReg.getBlocks().front();

    // Can't get DomTree for single block regions
    if (funcReg.hasOneBlock()) {
      llvm::outs() << "Region has only one Block. Cannot get DomTree for single block regions.\n";
      return failure();
    }
    // Get the data structure containg information about post-dominance.
    PostDominanceInfo postDomInfo;
    llvm::DominatorTreeBase<Block, true>& postDomTree = postDomInfo.getDomTree(&funcReg);

    // Print the post-dominance tree.
    mlir::DominanceInfoNode* rootNode = postDomTree.getRootNode();
    Block* rootBlockPostDom = rootNode->getBlock();
    //llvm::outs() << rootNode;
    PrintDomTree(rootNode, llvm::outs(), 0);

    PostDomTreeTraversal(rootNode, 0);

    // Find set of CFG edges (A,B) so A is NOT post-dominated by B.
    std::set<Block*> visitedSet;
    std::set<Edge*> edgeSet;
    // Memory for edges is allocated on the heap.
    CFGTraversal(&rootBlockCFG, &visitedSet, &edgeSet, &postDomInfo, postDomTree);

    // Process each edge from the set.
    for (Edge* edge : edgeSet) {
      // Find the least common ancesstor of A and B for each edge (A,B)
      DominanceInfoNode* LCA = edge->findLowestCommonAncestor();

      LCA->getBlock()->print(llvm::outs());
      llvm::outs() << "Edge processed.\n";
      
      if (LCA == edge->from->getIDom()) {
        // LCA is the parent of A node.
        // TO DO: (LCA, B] nodes are control dependent on A.

      }
      else if (LCA == edge->from) {
        // LCA is the A node.
        // TO DO: [A,B] nodes are control dependent on A. (?)
        
      }
      else {
        llvm::outs() << "Error in post-dominance tree.\n";
      }
    }

    llvm::outs() << "End of CDG analysis for FuncOp.\n";
    return success();
  }

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H