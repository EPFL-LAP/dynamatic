//===- CDGAnalysis.h - Exp. support for CDG analysis -------*- C++ -*-===//
//
// This file contains the function for CDG analysis.
//
//===-----------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H
#define EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include <queue>
#include <set>

using namespace mlir;

namespace dynamatic {
namespace experimental {

// Class that represents a node of the Control Dependency Graph.
// Should be used as a wrapper for type mlir::Block.
template <class NodeT>
class CDGNode {
  NodeT *BB;

  // This CDGNode is control dependent on its predecessors.
  SmallVector<CDGNode *, 4> predecessors;

  // The successors of this CDGNode are control dependent on this CDGNode.
  SmallVector<CDGNode *, 4> successors;

public:
  CDGNode(NodeT *BB) : BB(BB) {}

  NodeT *getBB() { return BB; }

  using iterator = typename SmallVector<CDGNode *, 4>::iterator;
  // predecessor iterator
  iterator beginPred() { return predecessors.begin(); }
  iterator endPred() { return predecessors.end(); }
  // successor iterator
  iterator beginSucc() { return successors.begin(); }
  iterator endSucc() { return successors.end(); }

  void addPredecessor(CDGNode *p) { predecessors.push_back(p); }
  void addSuccessor(CDGNode *s) { successors.push_back(s); }

  bool isLeaf() const { return successors.empty(); }
  bool isRoot() const { return predecessors.empty(); }

  size_t getNumSuccessors() const { return successors.size(); }
  void clearAllSuccessors() { successors.clear(); }
  size_t getNumPredecessors() const { return predecessors.size(); }
  void clearAllPredecessors() { predecessors.clear(); }
};

// CDG analysis function
// Function should return a pointer to the Entry node in CDG, returns nullptr if
// errror occurs.
CDGNode<Block> *CDGAnalysis(func::FuncOp funcOp, MLIRContext *ctx);

// Attach attributes to each BB terminator Operation, needed for testing.
void CDGTraversal(CDGNode<Block> *node, std::set<Block *> &visitedSet);

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H