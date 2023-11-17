//===- Graph.h - Represents a graph -----------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Graph class represents a graph composed of nodes and edges with the
// purpose of being displayed in Godot.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_VISUAL_DATAFLOW_GRAPH_H
#define DYNAMATIC_VISUAL_DATAFLOW_GRAPH_H

#include "GraphEdge.h"
#include "GraphNode.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dynamatic {
namespace experimental {
namespace visual_dataflow {

using CycleNb = int;
using GraphId = int;

/// State of an edge of the graph
enum State { UNDEFINED, READY, EMPTY, VALID, VALID_READY };

/// Implements the logic to create and update a Graph
class Graph {

public:
  /// Constructs a graph
  Graph() = default;
  /// Adds an edge to the graph
  void addEdge(GraphEdge edge);
  /// Adds a node to the graph
  void addNode(GraphNode node);
  /// Retrieves a node based on a given node identifier
  LogicalResult getNode(NodeId &id, GraphNode &result);
  /// Based on information about an edge, retrieves the corresponding edge
  /// identifier
  LogicalResult
  getEdgeId(std::pair<std::pair<NodeId, int>, std::pair<NodeId, int>> &edgeInfo,
            EdgeId &edgeId);
  /// Given a specific clock cycle, adds a pair (edge, state) to the map
  void addEdgeState(CycleNb cycle, EdgeId edgeId, State state);
  /// Returns all the Nodes in the Graph
  std::map<NodeId, GraphNode> getNodes();
  /// Returns all the edges in the Graph
  std::vector<GraphEdge> getEdges();

  std::map<CycleNb, std::map<EdgeId, State>> getCycleEdgeStates();

private:
  /// Edges of the graph
  std::vector<GraphEdge> edges;
  /// Nodes of the graph mapped with their corresponding node identifier
  std::map<NodeId, GraphNode> nodes;
  /// State of each edge given a specific clock cycle
  std::map<CycleNb, std::map<EdgeId, State>> cycleEdgeStates;
  /// Map of the edges of the graph :
  /// ((src node id, outPort number), (dest node id, inPort number)) -> edge id
  std::map<std::pair<std::pair<NodeId, int>, std::pair<NodeId, int>>, EdgeId>
      mapEdges;
};

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_VISUAL_DATAFLOW_GRAPH_H
