#include "experimental/Support/CDGAnalysis.h"

using namespace dynamatic::experimental;

namespace {

  // Helper struct for CDG analysis that represents the edge (A,B) in CFG.
  struct CFGEdge {
    DominanceInfoNode *from; // A
    DominanceInfoNode *to;   // B

    CFGEdge(DominanceInfoNode *from, DominanceInfoNode *to)
    : from(from), to(to){}

    // Finds the least common ancesstor (LCA) in post-dominator tree 
    // for nodes A and B of a CFG edge (A,B).
    DominanceInfoNode *findLCAInPostDomTree() {
      std::set<DominanceInfoNode *> ancestorsA;
      unsigned level;

      // Traverse upwards from node A and store its ancestors in the set.
      level = from->getLevel();
      for (DominanceInfoNode *curr = from; level > 0;
          --level, curr = curr->getIDom()) {
        ancestorsA.insert(curr);
      }

      // Traverse upwards from node B until a common ancestor is found.
      level = to->getLevel();
      for (DominanceInfoNode *curr = to; level > 0;
          --level, curr = curr->getIDom()) {
        if (ancestorsA.find(curr) != ancestorsA.end()) {
          // Lowest common ancestor
          return curr;
        }
      }

      return nullptr;
    }
  };

}

// Preorder post-dominance tree traversal
static void PostDomTreeTraversal(DominanceInfoNode *node, unsigned level) {
  if (!node || level < 0)
    return;

  // visit

  // end of visit

  for (typename DominanceInfoNode::iterator i = node->begin(),
                                            end = node->end();
       i != end; ++i) {
    PostDomTreeTraversal(*i, level + 1);
  }
}

// Traversal of the CFG that creates set of graph edges (A,B) so A is NOT
// post-dominated by B.
static void CFGTraversal(Block *rootBlock, std::set<Block *> *visitedSet, 
                         std::set<CFGEdge *> *edgeSet,
                         PostDominanceInfo *postDomInfo,
                         llvm::DominatorTreeBase<Block, true> &postDomTree) {
  if (!rootBlock) {
    return;
  }
  Block *curr = rootBlock;

  // Mark current node as visited.
  visitedSet->insert(curr);

  // end of visit

  for (Block *successor : curr->getSuccessors()) {
    // Check if curr is not post-dominated by his successor.
    if (postDomInfo->properlyPostDominates(successor, curr)) {
      // Every node must be marked as visited for the CDG construction purposes.
      // We will use this set to easly traverse each node and create corresponing CDGNode object.
      visitedSet->insert(successor);
      continue;
    }

    // Form the edge struct and add it to the set.
    DominanceInfoNode *succPostDomNode = postDomTree.getNode(successor);
    DominanceInfoNode *currPostDomNode = postDomTree.getNode(curr);

    CFGEdge *edge = new CFGEdge(currPostDomNode, succPostDomNode);
    edgeSet->insert(edge);

    // Check if successor is visited.
    if (visitedSet->find(successor) != visitedSet->end()) {
      // Node is already visited.
      continue;
    }

    CFGTraversal(successor, visitedSet, edgeSet, postDomInfo, postDomTree);
  }
}

// CDG traversal function
static void CDGTraversal(CDGNode<Block> *node, std::set<Block*> &visitedSet) {
  if (!node) return;

  // visit node
  visitedSet.insert(node->getBB());

  // end visit
  
  for (auto it = node->beginSucc(); it != node->endSucc(); ++it) {
    CDGNode<Block>* successor = *it;
    
    if (visitedSet.find(successor->getBB()) != visitedSet.end()) {
      // Successor is already visited.
      continue;
    }

    CDGTraversal(successor, visitedSet);
  }
}

CDGNode<Block>* dynamatic::experimental::CDGAnalysis(func::FuncOp funcOp,
                                                   MLIRContext *ctx) {

  Region &funcReg = funcOp.getRegion();
  Block &rootBlockCFG = funcReg.getBlocks().front();

  // Can't get DomTree for single block regions
  if (funcReg.hasOneBlock()) {
    llvm::outs() << "Region has only one Block. Cannot get DomTree for single "
                    "block regions.\n";
    return nullptr;
  }
  // Get the data structure containg information about post-dominance.
  PostDominanceInfo postDomInfo;
  llvm::DominatorTreeBase<Block, true> &postDomTree =
      postDomInfo.getDomTree(&funcReg);

  // Find set of CFG edges (A,B) so A is NOT post-dominated by B.
  std::set<Block *> visitedSet;
  std::set<CFGEdge *> edgeSet;
  // Memory for edges is allocated on the heap.
  CFGTraversal(&rootBlockCFG, &visitedSet, &edgeSet, &postDomInfo, postDomTree);

  std::unordered_map<Block*, CDGNode<Block>*> blockToCDGNodeMap;
  for (Block* block : visitedSet) {
    blockToCDGNodeMap[block] = new CDGNode<Block>(block);
  }

  // Process each edge from the set.
  for (CFGEdge *edge : edgeSet) {
    // Find the least common ancesstor (LCA) in post-dominator tree 
    // of A and B for each CFG edge (A,B).
    DominanceInfoNode *LCA = edge->findLCAInPostDomTree();

    Block *controlBlock = edge->from->getBlock(); /*A*/
    CDGNode<Block>* controlNode = blockToCDGNodeMap[controlBlock];

    // LCA = parent(A) or LCA = A.
    // All nodes on the path (LCA, B] are control dependant on A.

    for (DominanceInfoNode* curr = edge->to /*B*/; curr != LCA; curr = curr->getIDom() /*parent*/) {
      Block* dependantBlock = curr->getBlock();
      CDGNode<Block>* dependantNode = blockToCDGNodeMap[dependantBlock];

      dependantNode->addPredecessor(controlNode);
      controlNode->addSuccessor(dependantNode);
    } // for end
  } // process edge end

  // Deallocate memory for CFGEdge objects.
  for (auto edge : edgeSet) {
      delete edge;
  }
  edgeSet.clear();

  // Entry point in the CDG.
  CDGNode<Block>* entryCDGNode = new CDGNode<Block>(nullptr);
  // Connect all detached root CDG nodes to the entry CDG node.
  for (auto const& pair : blockToCDGNodeMap) {
    Block* key = pair.first;
    CDGNode<Block>* node = pair.second;
    
    if (node->isRoot()) {
      entryCDGNode->addSuccessor(node);
      node->addPredecessor(entryCDGNode);
    }
  }

  return entryCDGNode;
} // CDGAnalysis end