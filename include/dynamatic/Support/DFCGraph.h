//===------------- DFCGraph.h - Dataflow circuit graph ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A read-only directed graph view of (part of) a dataflow circuit: nodes are
// units (MLIR operations) and edges are channels (MLIR values).
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_DFCGRAPH_H
#define DYNAMATIC_SUPPORT_DFCGRAPH_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <vector>

namespace dynamatic {

class DFCGraph {
public:
  using NodeId = unsigned;
  using EdgeId = unsigned;

  /// A vertex of the graph (unit of the circuit). Neighbor pointers are stored
  /// so that llvm::GraphTraits can walk the graph from a node alone without
  /// having to traverse edges
  struct Node {
    mlir::Operation *unit;
    NodeId id;
    llvm::SmallVector<EdgeId, 4> outEdges;
    llvm::SmallVector<EdgeId, 4> inEdges;
    llvm::SmallVector<const Node *, 4> succs;
  };

  /// An edge (channel) from vertex (unit) `src` to vertex (unit) `dst`.
  struct Edge {
    NodeId src;
    NodeId dst;
    mlir::Value channel;
  };

  /// Builds the graph from any range of units and any range of channels.
  /// Channels for which `skipChannel` returns true are excluded.
  template <typename UnitRange, typename ChannelRange>
  static DFCGraph fromUnitsAndChannels(
      const UnitRange &units, const ChannelRange &channels,
      llvm::function_ref<bool(mlir::OpOperand &)> skipChannel = nullptr) {
    llvm::SmallVector<mlir::Operation *> unitVec(llvm::adl_begin(units),
                                                 llvm::adl_end(units));
    llvm::SmallVector<mlir::Value> channelVec(llvm::adl_begin(channels),
                                              llvm::adl_end(channels));
    DFCGraph graph;
    graph.build(unitVec, channelVec);
    return graph;
  }

  DFCGraph(const DFCGraph &) = delete;
  DFCGraph &operator=(const DFCGraph &) = delete;
  DFCGraph(DFCGraph &&) = default;
  DFCGraph &operator=(DFCGraph &&) = default;

  unsigned numNodes() const { return nodes.size(); }
  unsigned numEdges() const { return edges.size(); }
  const Node &getNode(NodeId id) const { return nodes[id]; }
  const Edge &getEdge(EdgeId id) const { return edges[id]; }
  llvm::ArrayRef<Node> getNodes() const { return nodes; }
  llvm::ArrayRef<Edge> getEdges() const { return edges; }

  /// Returns the node of a unit, if the unit is in the graph.
  std::optional<NodeId> lookup(mlir::Operation *unit) const;
  /// Returns the edge of a channel, if the channel is in the graph.
  std::optional<EdgeId> lookup(mlir::Value channel) const;

  /// "Ghost"" entry node with an edge to every node, so that algorithms that
  /// start from a single entry (e.g. scc_iterator that performs DFS) can reach
  /// the whole graph.
  /// It is not part of getNodes()/getEdges() and has no incoming edges, so it
  /// is never on a cycle.
  const Node *getRoot() const { return &root; }

  /// SCC ID of a unit. SCCs are computed once when the graph is built.
  unsigned getSCC(NodeId node) const { return sccOf[node]; }

  /// A channel lies on some cycle iff its producer and consumer are in the
  /// same SCC.
  bool isOnCycle(EdgeId edge) const {
    const Edge &e = edges[edge];
    return sccOf[e.src] == sccOf[e.dst];
  }

  /// A channel lies on some cycle iff its producer and consumer are in the
  /// same SCC.
  bool isOnCycle(mlir::Value channel) const {
    std::optional<EdgeId> edge = lookup(channel);
    assert(edge && "channel is not in the graph");
    return isOnCycle(*edge);
  }

private:
  DFCGraph() = default;

  void build(llvm::ArrayRef<mlir::Operation *> units,
             llvm::ArrayRef<mlir::Value> channels);
  void addUnits(llvm::ArrayRef<mlir::Operation *> units);
  void addEdge(NodeId src, NodeId dst, mlir::Value channel);
  /// Called once all units and channels are added: links the ghost root to
  /// every node and computes SCCs.
  void finalize();
  void computeSCCs();

  std::vector<Node> nodes;
  std::vector<Edge> edges;
  llvm::DenseMap<mlir::Operation *, NodeId> unitToNode;
  llvm::DenseMap<mlir::Value, EdgeId> channelToEdge;
  /// SCC ID of each node, indexed by NodeId.
  llvm::SmallVector<unsigned> sccOf;
  Node root{/*unit=*/nullptr, /*id=*/~0u, {}, {}, {}};
};

} // namespace dynamatic

namespace llvm {
template <>
struct GraphTraits<const dynamatic::DFCGraph *> {
  using NodeRef = const dynamatic::DFCGraph::Node *;
  using ChildIteratorType =
      SmallVectorImpl<const dynamatic::DFCGraph::Node *>::const_iterator;

  static NodeRef getEntryNode(const dynamatic::DFCGraph *graph) {
    return graph->getRoot();
  }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->succs.begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->succs.end(); }
};
} // namespace llvm

#endif // DYNAMATIC_SUPPORT_DFCGRAPH_H
