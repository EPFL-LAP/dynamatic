#include "experimental/Support/CDGAnalysis.h"

using namespace dynamatic::experimental;

template <class NodeT> 
CDGNode<NodeT>::CDGNode(NodeT* BB) {
  this->BB = BB;
}

template <class NodeT>
NodeT* CDGNode<NodeT>::getBB() {
  return BB;
}

// Preorder post-dominance tree traversal
static void PostDomTreeTraversal(DominanceInfoNode *node, unsigned level) {
  if (!node || level < 0)
    return;

  // visit
  DominanceInfoNode *iPostDomInf =
      node->getIDom(); // parent(node) in post dom tree

  // Exit node (the only node on level 0) does not have immediate
  // post-dominator. Immediate post-dominator of nodes on level 1 is the exit
  // node.
  if (level > 1 && iPostDomInf) {
    Block *iPostDomBlock = iPostDomInf->getBlock();
    iPostDomBlock->print(llvm::outs());
  }

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

  // end visit

  for (Block *successor : curr->getSuccessors()) {
    // llvm::outs() << "----- Edge:\n";
    // curr->print(llvm::outs());
    // successor->print(llvm::outs());
    // llvm::outs() << "-----------\n";

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

    llvm::outs() << "----- Edge added to the set:\n";
    curr->print(llvm::outs());
    successor->print(llvm::outs());
    llvm::outs() << "----------------------------\n";

    // Check if successor is visited.
    if (visitedSet->find(successor) != visitedSet->end()) {
      // Node is already visited.
      continue;
    }

    CFGTraversal(successor, visitedSet, edgeSet, postDomInfo, postDomTree);
  }
}

CFGEdge::CFGEdge(DominanceInfoNode *from, DominanceInfoNode *to)
    : from(from), to(to){};

DominanceInfoNode *CFGEdge::findLCAInPostDomTree() {
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

// CDG traversal function
static void CDGTraversal(CDGNode<Block> *node, std::set<Block*> &visitedSet) {
  if (!node) return;

  // visit node
  visitedSet.insert(node->getBB());

  llvm::outs() << "--------- CDGNode:\n";
  if (node->getBB()) {
    node->getBB()->print(llvm::outs());
  }
  else {
    llvm::outs() << "<entry node>\n";
  }
  for (auto it = node->beginSucc(); it != node->endSucc(); ++it) {
    CDGNode<Block>* successor = *it;

    llvm::outs() << "\t---- Child:";
    successor->getBB()->print(llvm::outs());
  }
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

// CDG analysis function
LogicalResult dynamatic::experimental::CDGAnalysis(func::FuncOp funcOp,
                                                   MLIRContext *ctx) {

  Region &funcReg = funcOp.getRegion();
  Block &rootBlockCFG = funcReg.getBlocks().front();

  // Can't get DomTree for single block regions
  if (funcReg.hasOneBlock()) {
    llvm::outs() << "Region has only one Block. Cannot get DomTree for single "
                    "block regions.\n";
    return failure();
  }
  // Get the data structure containg information about post-dominance.
  PostDominanceInfo postDomInfo;
  llvm::DominatorTreeBase<Block, true> &postDomTree =
      postDomInfo.getDomTree(&funcReg);

  // Print the post-dominance tree.
  mlir::DominanceInfoNode *rootNode = postDomTree.getRootNode();
  Block *rootBlockPostDom = rootNode->getBlock();
  // llvm::outs() << rootNode;
  PrintDomTree(rootNode, llvm::outs(), 0);

  PostDomTreeTraversal(rootNode, 0);

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

    LCA->getBlock()->print(llvm::outs());
    llvm::outs() << "Edge processed.\n";

    Block *controlBlock = edge->from->getBlock(); /*A*/
    CDGNode<Block>* controlNode = blockToCDGNodeMap[controlBlock];

    //if (LCA == edge->from->getIDom()) {
    // LCA is the parent of A node.
    // All nodes on the path (LCA, B] are control dependant on A.

    for (DominanceInfoNode* curr = edge->to /*B*/; curr != LCA; curr = curr->getIDom() /*parent*/) {
      Block* dependantBlock = curr->getBlock();
      CDGNode<Block>* dependantNode = blockToCDGNodeMap[dependantBlock];

      dependantNode->addPredecessor(controlNode);
      controlNode->addSuccessor(dependantNode);
    } // for end

    //}
    if (LCA == edge->from) {
      // LCA is the A node.
      // All nodes on the path [LCA, B] are control dependent on A.
      // Note that A can be control dependent on itself (loop dependency).

      // TO DO (?)
    }
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

  // Print the CDG.
  visitedSet.clear();
  CDGTraversal(entryCDGNode, visitedSet);

  llvm::outs() << "End of CDG analysis for FuncOp.\n";
  return success();
} // CDGAnalysis end