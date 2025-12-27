//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A generic dataflow graph representation for circuit analysis & optimization.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_DATAFLOWGRAPH_DATAFLOWSUBGRAPHBASE_H
#define DYNAMATIC_SUPPORT_DATAFLOWGRAPH_DATAFLOWSUBGRAPHBASE_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>

namespace dynamatic {

/// NOTE: No current implementation differentiates between intra-BB and inter-BB
/// edges. Right now, it's quite useful for visualizing the graph in GraphViz.
enum class DataflowGraphEdgeType {
  INTRA_BB, // <-- Edge within the same basic block.
  INTER_BB, // <-- Edge between different basic blocks.
};

template <typename T>
struct DataflowGraphNode {
  T op;      // <-- The underlying Operation.
  size_t id; // <-- Unique id in the nodes vector to help with traversal.

  DataflowGraphNode(T op, size_t id) : op(op), id(id) {}
};

template <typename T>
struct DataflowGraphEdge {
  size_t srcId;
  size_t dstId;

  T channel;
  DataflowGraphEdgeType type;

  DataflowGraphEdge(
      size_t srcId, size_t dstId, T channel,
      DataflowGraphEdgeType type = DataflowGraphEdgeType::INTRA_BB)
      : srcId(srcId), dstId(dstId), channel(channel), type(type) {}
};

/// Abstract base class for dataflow graphs used in circuit analysis &
/// optimization. It does the heavy lifting of managing the graph structure and
/// traversal. Inheriting classes need to implement type-specific methods to
/// determine what constitutes a fork and join. They then can add custom logic
/// to enumerate:
///     - Reconvergent paths from acyclic graphs
///     - Synchronizing paths from Choice-Free-Circuits (CFCs)

template <typename NodeType, typename EdgeType>
class DataflowSubgraphBase {
public:
  virtual ~DataflowSubgraphBase() = default;

  /// Virtual Methods ///

  virtual bool isForkNode(size_t nodeId) const = 0;
  virtual bool isJoinNode(size_t nodeId) const = 0;

  virtual std::string getNodeLabel(size_t nodeId) const = 0;
  virtual std::string getNodeDotId(size_t nodeId) const = 0;

  /// Getters ///

  handshake::FuncOp getFuncOp() const { return funcOp; }

  const std::vector<DataflowGraphNode<NodeType>> &getNodes() const {
    return nodes;
  }
  const std::vector<DataflowGraphEdge<EdgeType>> &getEdges() const {
    return edges;
  }

  const std::vector<llvm::SmallVector<size_t, 4>> &getAdjList() const {
    return adjList;
  }

  const std::vector<llvm::SmallVector<size_t, 4>> &getRevAdjList() const {
    return revAdjList;
  }

protected:
  handshake::FuncOp funcOp;

  std::vector<DataflowGraphNode<NodeType>> nodes;
  std::vector<DataflowGraphEdge<EdgeType>> edges;

  std::vector<llvm::SmallVector<size_t, 4>> adjList;
  std::vector<llvm::SmallVector<size_t, 4>> revAdjList;

  size_t addNode(NodeType op) {
    size_t id = nodes.size();
    nodes.emplace_back(op, id);
    adjList.emplace_back();
    revAdjList.emplace_back();
    return id;
  }

  void addEdge(size_t srcId, size_t dstId, EdgeType channel,
               DataflowGraphEdgeType type = DataflowGraphEdgeType::INTRA_BB) {
    edges.emplace_back(srcId, dstId, channel, type);
    adjList[srcId].push_back(edges.size() - 1);
    revAdjList[dstId].push_back(edges.size() - 1);
  }
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_DATAFLOWGRAPH_DATAFLOWSUBGRAPHBASE_H