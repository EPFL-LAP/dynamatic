//===- ControlNetworkAnalysis.h - Loops of the control network --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analysis of the control network of a Handshake function: the graph formed by
// the function's operations and the control-only channels between them. Its
// purpose is to recover the loops of the original program (and their nesting)
// from the dataflow circuit, which LLVM's generic dominator tree and loop
// machinery provide once the control network is presented to them as a graph.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_ANALYSIS_CONTROLNETWORKANALYSIS_H
#define DYNAMATIC_ANALYSIS_CONTROLNETWORKANALYSIS_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/GenericLoopInfo.h"
#include <deque>

namespace dynamatic {

class ControlGraph;

/// Node of a function's control graph: one per non-memory operation, plus a
/// single virtual entry node (standing for no operation) connected to every
/// root so that the whole graph is dominated by one entry (required to build a
/// dominator tree over an otherwise multi-rooted graph).
///
/// Nodes are owned by the graph they belong to and only ever built by it.
class ControlNode {
public:
  /// Returns the operation this node stands for, or nullptr for the graph's
  /// virtual entry node.
  Operation *getOp() const { return op; }

  /// Returns the graph this node belongs to. Also required by the dominator
  /// tree.
  ControlGraph *getParent() const { return graph; }

  /// Returns the nodes this one has a control channel to.
  ArrayRef<ControlNode *> getSuccessors() const { return succs; }

  /// Returns the nodes that have a control channel to this one.
  ArrayRef<ControlNode *> getPredecessors() const { return preds; }

  /// Required by the dominator tree.
  void printAsOperand(raw_ostream &os, bool printType = false) const;

private:
  /// The graph is the only builder of nodes and edges.
  friend class ControlGraph;

  Operation *op = nullptr;
  ControlGraph *graph = nullptr;
  SmallVector<ControlNode *> succs;
  SmallVector<ControlNode *> preds;
};

/// The control graph of a Handshake function: its operations connected by the
/// control-only channels between them.
class ControlGraph {
public:
  using iterator = llvm::pointer_iterator<std::deque<ControlNode>::iterator>;

  /// Builds the control graph of 'funcOp'.
  explicit ControlGraph(handshake::FuncOp funcOp);

  /// The nodes point back at the graph owning them, which a copy would leave
  /// pointing at the original.
  ControlGraph(const ControlGraph &) = delete;
  ControlGraph &operator=(const ControlGraph &) = delete;

  /// The graph's nodes, the virtual entry node first.
  iterator begin() { return iterator(nodes.begin()); }
  iterator end() { return iterator(nodes.end()); }

  /// Returns the node standing for 'op', or nullptr if the operation is not
  /// part of the graph (memory interfaces are not).
  ControlNode *getNode(Operation *op) const { return nodeFor.lookup(op); }

  /// Returns the virtual entry node, which dominates every other node. Required
  /// by 'DomTreeNodeTraits'.
  ControlNode &front() { return nodes.front(); }

private:
  /// Appends a node standing for 'op' (null for the entry node) and returns it.
  ControlNode *addNode(Operation *op);

  /// Adds a control edge from 'src' to 'dst'.
  static void addEdge(ControlNode *src, ControlNode *dst);

  /// A deque for the stable addresses, as the nodes point to each other.
  std::deque<ControlNode> nodes;
  /// Maps each operation to its node.
  DenseMap<Operation *, ControlNode *> nodeFor;
};

} // namespace dynamatic

namespace llvm {
template <>
struct GraphTraits<dynamatic::ControlNode *> {
  using NodeRef = dynamatic::ControlNode *;
  using ChildIteratorType = ArrayRef<dynamatic::ControlNode *>::iterator;
  static NodeRef getEntryNode(NodeRef node) { return node; }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->getSuccessors().begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->getSuccessors().end();
  }
};

template <>
struct GraphTraits<Inverse<dynamatic::ControlNode *>> {
  using NodeRef = dynamatic::ControlNode *;
  using ChildIteratorType = ArrayRef<dynamatic::ControlNode *>::iterator;
  static NodeRef getEntryNode(Inverse<dynamatic::ControlNode *> inv) {
    return inv.Graph;
  }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->getPredecessors().begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->getPredecessors().end();
  }
};

template <>
struct GraphTraits<dynamatic::ControlGraph *>
    : public GraphTraits<dynamatic::ControlNode *> {
  using nodes_iterator = dynamatic::ControlGraph::iterator;
  static NodeRef getEntryNode(dynamatic::ControlGraph *graph) {
    return &graph->front();
  }
  static nodes_iterator nodes_begin(dynamatic::ControlGraph *graph) {
    return graph->begin();
  }
  static nodes_iterator nodes_end(dynamatic::ControlGraph *graph) {
    return graph->end();
  }
};

template <>
struct GraphTraits<const DomTreeNodeBase<dynamatic::ControlNode> *> {
  using NodeRef = const DomTreeNodeBase<dynamatic::ControlNode> *;
  using ChildIteratorType =
      DomTreeNodeBase<dynamatic::ControlNode>::const_iterator;
  static NodeRef getEntryNode(NodeRef node) { return node; }
  static ChildIteratorType child_begin(NodeRef node) { return node->begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node->end(); }
};
} // namespace llvm

namespace dynamatic {

/// Concrete loop type for 'LoopInfoBase' over the control graph.
class ControlLoop : public llvm::LoopBase<ControlNode, ControlLoop> {
public:
  ControlLoop() = default;

  /// Returns the deepest nesting depth reachable from this loop through its own
  /// descendants (i.e. 'getLoopDepth()' itself when the loop is innermost).
  ///
  /// The result is not cached: every call walks all of the loops nested inside
  /// this one, making it linear in their number (i.e. in the nesting level for
  /// a perfect nest).
  unsigned getMaxLoopDepth() const;

private:
  friend class llvm::LoopBase<ControlNode, ControlLoop>;
  friend class llvm::LoopInfoBase<ControlNode, ControlLoop>;

  explicit ControlLoop(ControlNode *node)
      : llvm::LoopBase<ControlNode, ControlLoop>(node) {}
};

/// Analysis recovering the loops of a Handshake function from its control
/// network, along with their nesting.
class ControlNetworkAnalysis {
public:
  /// Builds the control graph of 'funcOp' and detects its loops.
  explicit ControlNetworkAnalysis(handshake::FuncOp funcOp);

  /// The dominator tree and loop info are built over the graph held here, so
  /// the analysis cannot be copied any more than the graph can.
  ControlNetworkAnalysis(const ControlNetworkAnalysis &) = delete;
  ControlNetworkAnalysis &operator=(const ControlNetworkAnalysis &) = delete;

  /// Returns every loop of the control network, outer loops before the loops
  /// they contain.
  SmallVector<ControlLoop *> getLoopsInPreorder() const;

  /// Returns the control graph's node for 'op', or nullptr if the operation is
  /// not part of the control graph (memory interfaces are not).
  ControlNode *getNode(Operation *op) const { return graph.getNode(op); }

private:
  ControlGraph graph;
  llvm::DominatorTreeBase<ControlNode, /*IsPostDom=*/false> domTree;
  llvm::LoopInfoBase<ControlNode, ControlLoop> loopInfo;
};

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_CONTROLNETWORKANALYSIS_H
