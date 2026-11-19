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

namespace dynamatic {
namespace buffer {

/// A set of ArchBB's pointers with a deterministic iteration order.
using ArchSet = mlir::SetVector<dynamatic::experimental::ArchBB *>;
/// A set of basic block IDs with a deterministic iteration order.
using BBSet = mlir::SetVector<unsigned>;

/// Represents a CFDFC i.e., a set of control-free units and channels from a
/// dataflow circuit accompanied by the number of times it was executed.
struct CFDFC {
  /// The list of basic blocks that make up the CFDFC.
  mlir::SetVector<unsigned> cycle;
  /// Units (i.e., MLIR operations) in the CFDFC.
  mlir::SetVector<Operation *> units;
  /// Channels (i.e., MLIR values) in the CFDFC.
  mlir::SetVector<Value> channels;
  /// Backedges in the CFDFC.
  mlir::SetVector<Value> backedges;
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

  /// Returns whether the channel between the two blocks has a corresponding
  /// edge in this CFDFC's CFG cycle.
  bool isCFGCompliant(unsigned srcBB, unsigned dstBB) const;

  // Determines whether the channel is a "CFDFC backedge" i.e., the first
  // channel along a sequence of backedges from a source block to a destination
  // block. The distinction is important for the buffer placement MILP, which
  // uses backedges to determine where to insert "tokens" in the circuit.
  static bool isCFDFCBackedge(Value val);

private:
  /// Builds the block cycle from the selected archs and collects the units
  /// belonging to it. Channels are collected afterwards by one of the
  /// collect functions.
  CFDFC(handshake::FuncOp funcOp, ArchSet &archs, unsigned numExec);

  /// Collects the channels and backedges of the units using the traditional
  /// channel selection.
  void collectChannels();

  /// Collects the channels and backedges of the units using the FTD-specific
  /// channel selection.
  void collectChannelsFTD(const DenseSet<Value> &backwardChannels);
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

/// A read-only directed graph view of a CFDFC: nodes are the CFDFC's units and
/// edges are the CFDFC's channels. SCCs are computed once on construction.
///
/// Not every unit of a CFDFC lies on a cycle (e.g., sources, sinks, or units
/// whose results leave the CFDFC), so the graph is in general not strongly
/// connected. A ghost root with an edge to every node makes the whole graph
/// reachable from a single entry node and allows us to use LLVMs SCC
/// infrastructure.
class CFDFCGraph {
public:
  using NodeId = unsigned;
  using EdgeId = unsigned;

  /// A vertex of the graph (unit of the cfdfc). Successor pointers are
  /// stored so that llvm::GraphTraits can walk the graph from a node alone.
  struct Node {
    Operation *unit;
    NodeId id;
    mlir::SmallVector<EdgeId, 4> outEdges;
    mlir::SmallVector<EdgeId, 4> inEdges;
    mlir::SmallVector<const Node *, 4> succs;
  };

  /// An edge (channel) from vertex (unit) `src` to vertex (unit) `dst`.
  struct Edge {
    NodeId src;
    NodeId dst;
    Value channel;
  };

  /// Builds the graph from the units and channels of the CFDFC. The graph does
  /// not reference the CFDFC after construction.
  explicit CFDFCGraph(const CFDFC &cfdfc);

  CFDFCGraph(const CFDFCGraph &) = delete;
  CFDFCGraph &operator=(const CFDFCGraph &) = delete;
  CFDFCGraph(CFDFCGraph &&) = default;
  CFDFCGraph &operator=(CFDFCGraph &&) = default;

  unsigned numNodes() const { return nodes.size(); }
  unsigned numEdges() const { return edges.size(); }
  const Node &getNode(NodeId id) const { return nodes[id]; }
  const Edge &getEdge(EdgeId id) const { return edges[id]; }
  mlir::ArrayRef<Node> getNodes() const { return nodes; }
  mlir::ArrayRef<Edge> getEdges() const { return edges; }

  /// Returns the node of a unit, if the unit is in the graph.
  std::optional<NodeId> lookup(Operation *unit) const;
  /// Returns the edge of a channel, if the channel is in the graph.
  std::optional<EdgeId> lookup(Value channel) const;

  /// Ghost entry node with an edge to every node, so that algorithms starting
  /// from a single entry (e.g., the DFS in llvm::scc_iterator) reach the whole
  /// graph. It is not part of getNodes()/getEdges() and has no incoming edges,
  /// so it is never on a cycle.
  const Node *getRoot() const { return &root; }

  /// Returns the SCC ID of a node.
  unsigned getSCC(NodeId node) const { return sccOf[node]; }

  /// A channel lies on some cycle iff its producer and consumer are in the
  /// same SCC.
  bool isOnCycle(EdgeId edge) const {
    const Edge &e = edges[edge];
    return sccOf[e.src] == sccOf[e.dst];
  }

  /// A channel lies on some cycle iff its producer and consumer are in the
  /// same SCC.
  bool isOnCycle(Value channel) const {
    std::optional<EdgeId> edge = lookup(channel);
    assert(edge && "channel is not in the graph");
    return isOnCycle(*edge);
  }

private:
  void addUnits(mlir::ArrayRef<Operation *> units);
  void addChannels(mlir::ArrayRef<Value> channels);
  void addEdge(NodeId src, NodeId dst, Value channel);
  /// Links the ghost root to every node and computes SCCs. Must be called once
  /// all units and channels are added.
  void finalize();
  void computeSCCs();

  std::vector<Node> nodes;
  std::vector<Edge> edges;
  mlir::DenseMap<Operation *, NodeId> unitToNode;
  mlir::DenseMap<Value, EdgeId> channelToEdge;
  /// SCC ID of each node, indexed by NodeId.
  mlir::SmallVector<unsigned> sccOf;
  Node root{/*unit=*/nullptr, /*id=*/~0u, {}, {}, {}};
};

} // namespace buffer
} // namespace dynamatic

namespace llvm {
template <>
struct GraphTraits<const dynamatic::buffer::CFDFCGraph *> {
  using NodeRef = const dynamatic::buffer::CFDFCGraph::Node *;
  using ChildIteratorType = SmallVectorImpl<NodeRef>::const_iterator;

  static NodeRef getEntryNode(const dynamatic::buffer::CFDFCGraph *graph) {
    return graph->getRoot();
  }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->succs.begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->succs.end(); }
};
} // namespace llvm

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H
