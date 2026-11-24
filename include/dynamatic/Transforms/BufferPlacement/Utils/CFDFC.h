//===- CFDFC.h - Control-Free DataFlow Circuit ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares data structures and functions to extract and create CFDFCs
// (Choice-Free DataFlow Circuits) from a description of a Handshake function's
// archs and basic blocks.
//
// In this context, an arch is understood as an "edge" between two basic blocks.
// There exists an arch in a function between basic block X and Y if and only if
// there exists an operation in X that returns a result used by an operation in
// Y.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/StdProfiler.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"
#include <optional>
#include <vector>

namespace dynamatic {
namespace buffer {

/// A set of ArchBB's pointers with a deterministic iteration order.
using ArchSet = mlir::SetVector<dynamatic::experimental::ArchBB *>;
/// A set of basic block IDs with a deterministic iteration order.
using BBSet = mlir::SetVector<unsigned>;
/// Index of a node (unit) in a CFDFC.
using CFDFCNodeId = unsigned;
/// Index of an edge (channel) in a CFDFC.
using CFDFCEdgeId = unsigned;

/// A unit of a CFDFC together with the CFDFC channels that enter and leave it.
/// Nodes are owned by their CFDFC and only ever built by it.
class CFDFCNode {
public:
  /// Returns the unit this node stands for, or nullptr for the CFDFC's ghost
  /// entry node.
  Operation *getUnit() const { return unit; }

  /// Returns the index of the node in its CFDFC. The ghost entry node's index
  /// is the number of nodes of the CFDFC.
  CFDFCNodeId getId() const { return id; }

  /// Returns the CFDFC edges entering the node.
  ArrayRef<CFDFCEdgeId> getInEdges() const { return inEdges; }

  /// Returns the CFDFC edges leaving the node.
  ArrayRef<CFDFCEdgeId> getOutEdges() const { return outEdges; }

  /// Returns the nodes this one has a CFDFC edge to.
  ArrayRef<const CFDFCNode *> getSuccessors() const { return succs; }

  /// Returns the nodes that have a CFDFC edge to this one.
  ArrayRef<const CFDFCNode *> getPredecessors() const { return preds; }

private:
  /// The CFDFC is the only builder of nodes and edges.
  friend struct CFDFC;

  /// Builds the CFDFC's ghost entry node.
  CFDFCNode() = default;
  CFDFCNode(Operation *unit, CFDFCNodeId id) : unit(unit), id(id) {}

  Operation *unit = nullptr;
  CFDFCNodeId id = 0;
  SmallVector<CFDFCEdgeId, 4> inEdges;
  SmallVector<CFDFCEdgeId, 4> outEdges;
  /// Derived from the edges by the owning CFDFC, which rebuilds them whenever
  /// the nodes may have moved in memory.
  SmallVector<const CFDFCNode *, 4> succs;
  SmallVector<const CFDFCNode *, 4> preds;
};

/// A channel of a CFDFC, from the node of its producer to the node of its
/// consumer. Edges are owned by their CFDFC and only ever built and modified by
/// it.
class CFDFCEdge {
public:
  /// Returns the channel this edge stands for.
  Value getChannel() const { return channel; }

  /// Returns the node of the edge's producer.
  CFDFCNodeId getSrc() const { return src; }

  /// Returns the node of the edge's consumer.
  CFDFCNodeId getDst() const { return dst; }

  /// Returns whether the channel is a "CFDFC backedge" (see
  /// CFDFC::isCFDFCBackedge).
  bool isBackedge() const { return backedge; }

private:
  /// The CFDFC is the only builder of nodes and edges.
  friend struct CFDFC;

  CFDFCEdge(Value channel, CFDFCNodeId src, CFDFCNodeId dst, bool backedge)
      : channel(channel), src(src), dst(dst), backedge(backedge) {}

  Value channel;
  CFDFCNodeId src;
  CFDFCNodeId dst;
  bool backedge;
  bool ignoredForThroughput = false;
};

/// Represents a CFDFC i.e., a set of control-free units and channels from a
/// dataflow circuit accompanied by the number of times it was executed.
/// The units and channels form a directed graph.
struct CFDFC {
  /// The list of basic blocks that make up the CFDFC.
  mlir::SetVector<unsigned> cycle;
  /// Number of executions of the CFDFC.
  unsigned numExecs;

  /// (Available after placement) The achieved throughput after the placement
  double throughput = 0.0;
  /// (Available after placement) The number of tokens held by each placed
  /// buffer.
  mlir::DenseMap<Operation *, double> bufferOccupancy;
  /// (Available after placement) The number of tokens per channel. After
  /// instantiating the buffers, this value is cleared and transferred to the
  /// buffer(s).
  mlir::DenseMap<Value, double> channelOccupancy;

  /// Removes a channel from the CFDFC. Its producer and consumer stay in the
  /// CFDFC. The order of the `channels` returned by `getChannels()` is not
  /// preservedby this method.
  void removeChannel(Value channel);

  /// Constructs a CFDFC from a set of selected archs and basic blocks in the
  /// function using the traditional channel selection.
  static CFDFC create(handshake::FuncOp funcOp, ArchSet &archs,
                      unsigned numExec);

  /// Constructs a CFDFC from a set of selected archs and basic blocks in the
  /// function using the FTD-specific channel selection, which additionally
  /// requires backedges to be specified.
  static CFDFC createFTD(handshake::FuncOp funcOp, ArchSet &archs,
                         unsigned numExec,
                         const DenseSet<Value> &backwardChannels);

  //===--------------------------------------------------------------------===//
  // Graph access
  //===--------------------------------------------------------------------===//

  /// Returns the nodes of the CFDFC, indexed by their ID.
  ArrayRef<CFDFCNode> getNodes() const { return graph.nodes; }
  /// Returns the edges (channels) of the CFDFC, indexed by their ID.
  ArrayRef<CFDFCEdge> getEdges() const { return graph.edges; }
  const CFDFCNode &getNode(CFDFCNodeId id) const { return graph.nodes[id]; }
  const CFDFCEdge &getEdge(CFDFCEdgeId id) const { return graph.edges[id]; }

  /// Returns the units of the CFDFC, in node order.
  auto getUnits() const {
    return llvm::map_range(
        graph.nodes, [](const CFDFCNode &node) { return node.getUnit(); });
  }
  /// Returns the channels of the CFDFC, in edge order.
  auto getChannels() const {
    return llvm::map_range(
        graph.edges, [](const CFDFCEdge &edge) { return edge.getChannel(); });
  }
  /// Returns the backedges of the CFDFC, in edge order. In FTD mode, see also
  /// getFTDBackwardEdges.
  auto getBackedges() const {
    return llvm::map_range(
        llvm::make_filter_range(
            graph.edges,
            [](const CFDFCEdge &edge) { return edge.isBackedge(); }),
        [](const CFDFCEdge &edge) { return edge.getChannel(); });
  }

  size_t getNumUnits() const { return graph.nodes.size(); }
  size_t getNumChannels() const { return graph.edges.size(); }
  size_t getNumBackedges() const;

  /// (FTD only) Backward channels between units of the CFDFC that are not
  /// channels of the CFDFC because they do not follow an edge of the CFDFC's
  /// block cycle. They are not part of the graph: their nodes do not list them.
  ArrayRef<CFDFCEdge> getFTDBackwardEdges() const {
    return graph.ftdBackwardEdges;
  }

  /// Returns the node of a unit, if the unit is in the CFDFC.
  std::optional<CFDFCNodeId> lookup(Operation *unit) const;
  /// Returns the edge of a channel, if the channel is in the CFDFC.
  std::optional<CFDFCEdgeId> lookup(Value channel) const;

  bool contains(Operation *unit) const { return lookup(unit).has_value(); }
  bool contains(Value channel) const { return lookup(channel).has_value(); }

  /// Returns whether the channel is a backedge of the CFDFC. Channels outside
  /// the CFDFC are not.
  bool isBackedge(Value channel) const;

  /// Returns the ghost root node, which is not part of getNodes(), with an edge
  /// to every node and no incoming edge. Algorithms that start from a single
  /// entry node (e.g., llvm::scc_iterator) reach the whole CFDFC from it.
  const CFDFCNode *getEntryNode() const { return &graph.entry; }

  //===--------------------------------------------------------------------===//
  // Graph modification
  //===--------------------------------------------------------------------===//

  /// Adds units placed on a channel of the CFDFC (e.g., buffers) to the CFDFC.
  /// The units must form a chain from the channel's producer to its original
  /// consumer: the first unit consumes `channel` and each unit's first result
  /// is the only input of the next unit or of the original consumer. The
  /// channel's edge then ends at the first unit, and the results of the units
  /// become new, non-backedge channels.
  void insertUnitsOnChannel(Value channel, ArrayRef<Operation *> newUnits);

  //===--------------------------------------------------------------------===//
  // Helpers
  //===--------------------------------------------------------------------===//

  /// Returns whether the channel between the two blocks has a corresponding
  /// edge in this CFDFC's CFG cycle.
  bool isCFGCompliant(unsigned srcBB, unsigned dstBB) const;

  // Determines whether the channel is a "CFDFC backedge" i.e., the first
  // channel along a sequence of backedges from a source block to a destination
  // block. The distinction is important for the buffer placement MILP, which
  // uses backedges to determine where to insert "tokens" in the circuit.
  static bool isCFDFCBackedge(Value val);

private:
  /// Storage of the CFDFC's graph. Nodes point to each other, so a copy
  /// rebuilds these pointers to point into the copy. Moves keep the nodes at
  /// the same address and need nothing special.
  struct Graph {
    Graph() = default;
    Graph(const Graph &other);
    Graph &operator=(const Graph &other);
    Graph(Graph &&) = default;
    Graph &operator=(Graph &&) = default;

    /// Appends a node for the unit, which must not be in the graph yet.
    CFDFCNodeId addNode(Operation *unit);
    /// Appends an edge for the channel, which must not be in the graph yet,
    /// between two existing nodes.
    CFDFCEdgeId addEdge(Value channel, CFDFCNodeId src, CFDFCNodeId dst,
                        bool isBackedge);
    /// Recomputes the nodes' successor and predecessor pointers and the entry
    /// node's successors from the edges. Must be called after any change to the
    /// CFDFC.
    void rebuild();

    /// A std::vector rather than a SmallVector so that a move never relocates
    /// the nodes.
    std::vector<CFDFCNode> nodes;
    std::vector<CFDFCEdge> edges;
    std::vector<CFDFCEdge> ftdBackwardEdges;
    DenseMap<Operation *, CFDFCNodeId> nodeOfUnit;
    DenseMap<Value, CFDFCEdgeId> edgeOfChannel;
    CFDFCNode entry;
  };

  /// Builds the block cycle from the selected archs and collects the units
  /// belonging to it. Channels are collected afterwards by one of the
  /// collect functions.
  CFDFC(handshake::FuncOp funcOp, ArchSet &archs, unsigned numExec);

  /// Adds a channel between two units of the CFDFC to the graph.
  void addChannel(Value channel, bool isBackedge);

  /// Collects the channels and backedges of the units using the traditional
  /// channel selection.
  void collectChannels();

  /// Collects the channels and backedges of the units using the FTD-specific
  /// channel selection.
  void collectChannelsFTD(const DenseSet<Value> &backwardChannels);

  Graph graph;
};

/// Represents a union of CFDFCs. Its blocks, units, channels, and backedges are
/// unions of the corresponding elements from all the CFDFCs that make it up. A
/// CFDFC union, unlike a regular CFDFC, does not contain a number of executions
/// since it does not make sense in the context of a union.
struct CFDFCUnion {
  /// The individual CFDFCs the union is made up of.
  SmallVector<CFDFC *> cfdfcs;
  /// The set basic blocks that make up the CFDFC union.
  mlir::SetVector<unsigned> blocks;
  /// Units (i.e., MLIR operations) in the CFDFC union.
  mlir::SetVector<Operation *> units;
  /// Channels (i.e., MLIR values) in the CFDFC union.
  mlir::SetVector<Value> channels;
  /// Backedges in the CFDFC union.
  mlir::SetVector<Value> backedges;

  /// Constructs the CFDFC union from an array of individual CFDFCs. The CFDFCs
  /// must outlive the union, which stores pointers to them.
  CFDFCUnion(ArrayRef<CFDFC *> cfdfcs);
};

/// Computes a set of CFDFC unions from a list of CFDFCs where each CFDFC in
/// each union shares at least one block with at least one other CFDFC of the
/// same union, and any two CFDFCs in different unions are completely disjoint.
/// Internally uses a disjoint-set data-structure for fast computation times and
/// low memory footprint.
void getDisjointBlockUnions(ArrayRef<CFDFC *> cfdfcs,
                            std::vector<CFDFCUnion> &unions);

/// Extracts the most frequently executed CFDFC from the Handshake function
/// described by the provided archs and basic blocks. The function internally
/// expresses the CFDFC extraction problem as an MILP that is solved bu Gurobi
/// (hence building the project with Gurobi is required to use this function).
/// On successfull extraction, succeeds and sets the last two arguments with,
/// respectively, the set of archs included in the extracted CFDFC and the
/// number of executions of the latter. When no CFDFC could be extracted,
/// succeeds but sets the number of executions to 0. On failure, and if
/// `milpStat` is not nullptr, the Gurobi status is saved in it.
LogicalResult extractCFDFC(handshake::FuncOp funcOp, ArchSet &archs, BBSet &bbs,
                           ArchSet &selectedArchs, unsigned &numExec,
                           CPSolver::SolverKind solverKind,
                           const std::string &logPath = "",
                           int *milpStat = nullptr);

} // namespace buffer
} // namespace dynamatic

namespace llvm {
template <>
struct GraphTraits<const dynamatic::buffer::CFDFCNode *> {
  using NodeRef = const dynamatic::buffer::CFDFCNode *;
  using ChildIteratorType =
      ArrayRef<const dynamatic::buffer::CFDFCNode *>::iterator;
  static NodeRef getEntryNode(NodeRef node) { return node; }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->getSuccessors().begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->getSuccessors().end();
  }
};

/// Walking a CFDFC starts from its ghost root, from which every nod eis
/// guaranteed ot be reachable.
template <>
struct GraphTraits<const dynamatic::buffer::CFDFC *>
    : public GraphTraits<const dynamatic::buffer::CFDFCNode *> {
  using nodes_iterator =
      pointer_iterator<ArrayRef<dynamatic::buffer::CFDFCNode>::iterator>;
  static NodeRef getEntryNode(const dynamatic::buffer::CFDFC *cfdfc) {
    return cfdfc->getEntryNode();
  }
  static nodes_iterator nodes_begin(const dynamatic::buffer::CFDFC *cfdfc) {
    return nodes_iterator(cfdfc->getNodes().begin());
  }
  static nodes_iterator nodes_end(const dynamatic::buffer::CFDFC *cfdfc) {
    return nodes_iterator(cfdfc->getNodes().end());
  }
  static unsigned size(const dynamatic::buffer::CFDFC *cfdfc) {
    return cfdfc->getNumUnits();
  }
};
} // namespace llvm

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H
