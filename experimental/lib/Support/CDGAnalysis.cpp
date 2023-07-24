#include <queue>
#include <set>
#include <stack>

#include "experimental/Support/CDGAnalysis.h"

using namespace dynamatic::experimental;

namespace {

/// @brief Helper struct for Control Dependence Graph (CDG) analysis that
/// represents a directed edge (A, B) in the Control Flow Graph (CFG).
struct CFGEdge {
  /// Node A of the CFG edge (A, B).
  DominanceInfoNode *a;
  /// Node B of the CFG edge (A, B).
  DominanceInfoNode *b;

  CFGEdge(DominanceInfoNode *a, DominanceInfoNode *b) : a(a), b(b) {}

  /// @brief Finds the Least Common Ancestor (LCA) in the Post-Dominator Tree
  /// for nodes A and B of a CFG edge (A, B).
  ///
  /// @return The Lowest Common Ancestor (LCA) in the Post-Dominator Tree for
  /// nodes A and B, or nullptr if no common ancestor is found.
  DominanceInfoNode *findLCAInPostDomTree() {
    std::set<DominanceInfoNode *> ancestorsA;

    // Traverse upwards from node A and store its ancestors in the set.
    unsigned level = a->getLevel();
    for (DominanceInfoNode *curr = a; level > 0;
         --level, curr = curr->getIDom())
      ancestorsA.insert(curr);

    // Traverse upwards from node B until a common ancestor is found.
    level = b->getLevel();
    for (DominanceInfoNode *curr = b; level > 0;
         --level, curr = curr->getIDom())
      if (ancestorsA.find(curr) != ancestorsA.end())
        // Lowest common ancestor
        return curr;

    return nullptr;
  }
};

/// @brief Traversal of the Control Flow Graph (CFG) that creates a set of graph
/// edges (A, B) where A is NOT post-dominated by B.
void cfgTraversal(Block &rootBlock, std::queue<CFGEdge *> &edgeSet,
                  PostDominanceInfo &postDomInfo,
                  llvm::DominatorTreeBase<Block, true> &postDomTree) {
  std::set<Block *> visitedSet;
  std::stack<Block *> blockStack;

  blockStack.push(&rootBlock);

  while (!blockStack.empty()) {
    Block *currBlock = blockStack.top();
    blockStack.pop();

    // Mark current node as visited.
    visitedSet.insert(currBlock);

    // end of visit

    for (Block *successor : currBlock->getSuccessors()) {
      // Check if curr is not post-dominated by his successor.
      if (postDomInfo.properlyPostDominates(successor, currBlock)) {
        // Every node must be marked as visited for the CDG construction
        // purposes. We will use this set to easly traverse each node and create
        // corresponing CDGNode object.
        visitedSet.insert(successor);
        continue;
      }

      // Form the edge struct and add it to the set.
      DominanceInfoNode *succPostDomNode = postDomTree.getNode(successor);
      DominanceInfoNode *currPostDomNode = postDomTree.getNode(currBlock);

      CFGEdge *edge = new CFGEdge(currPostDomNode, succPostDomNode);
      edgeSet.push(edge);

      // Check if successor is already visited.
      if (visitedSet.find(successor) != visitedSet.end())
        continue;

      blockStack.push(successor);
    }
  } // end while
}

} // namespace

DenseMap<Block *, BlockNeighbors *> *
dynamatic::experimental::cdgAnalysis(func::FuncOp &funcOp, MLIRContext &ctx) {
  DenseMap<Block *, BlockNeighbors *> *cdg =
      new DenseMap<Block *, BlockNeighbors *>();

  Region &funcReg = funcOp.getRegion();

  for (Block &block : funcReg.getBlocks())
    cdg->insert(std::make_pair(&block, new BlockNeighbors()));

  // Handling single block regions,
  // cannot get DomTree for single block regions.
  if (funcReg.hasOneBlock())
    return cdg;

  Block &rootBlockCFG = funcReg.getBlocks().front();
  // Get the data structure containg information about post-dominance.
  PostDominanceInfo postDomInfo;
  llvm::DominatorTreeBase<Block, true> &postDomTree =
      postDomInfo.getDomTree(&funcReg);

  // Find set of CFG edges (A,B) so A is NOT post-dominated by B.
  // Memory for edges is allocated on the heap.
  std::queue<CFGEdge *> edgeSet;
  cfgTraversal(rootBlockCFG, edgeSet, postDomInfo, postDomTree);

  // Process each edge from the set.
  while (!edgeSet.empty()) {
    CFGEdge *edge = edgeSet.front();
    edgeSet.pop();
    // Find the least common ancesstor (LCA) in post-dominator tree
    // of A and B for each CFG edge (A,B).
    DominanceInfoNode *lca = edge->findLCAInPostDomTree();

    Block *controlBlock = edge->a->getBlock();

    // LCA = parent(A) or LCA = A.
    // All nodes on the path (LCA, B] are control dependent on A.

    for (DominanceInfoNode *curr = edge->b; curr != lca;
         curr = curr->getIDom()) {
      Block *dependentBlock = curr->getBlock();

      (*cdg)[controlBlock]->successors.push_back(dependentBlock);
      (*cdg)[dependentBlock]->predecessors.push_back(controlBlock);
    } // for end

    // Deallocate memory for CFGEdge objects.
    delete edge;
  } // process edge end

  return cdg;
} // CDGAnalysis end
