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
namespace visual {

using CycleNb = int;
using GraphId = int;
using Data = std::string;

/// State of an edge of the graph
enum State { UNDEFINED, ACCEPT, IDLE, STALL, TRANSFER };

struct BB {
  std::vector<float> boundries;
  std::string label;
  std::pair<float, float> labelPosition;
  std::pair<float, float> labelSize;
};

/// Stores channel state transitions as a map from edge IDs to corresponding
/// state transition information (and data, when relevant).
using ChannelTransitions = std::map<EdgeId, std::pair<State, Data>>;

/// Stores the set of state transitionss at each cycle, mapping each cycle
/// number to the set of channel state transitions that occur during it.
using CycleTransitions = std::map<CycleNb, ChannelTransitions>;

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
  LogicalResult getEdgeId(
      std::pair<std::pair<NodeId, size_t>, std::pair<NodeId, size_t>> &edgeInfo,
      EdgeId &edgeId);
  /// Given a specific clock cycle, adds a pair (edge, state) to the map
  void addEdgeState(CycleNb cycle, EdgeId edgeId, State state,
                    const Data &data);
  /// Returns all the Nodes in the Graph
  std::map<NodeId, GraphNode> getNodes();
  /// Returns all the edges in the Graph
  std::vector<GraphEdge> getEdges();

  CycleTransitions &getCycleEdgeStates();

  void dupilcateEdgeStates(CycleNb from, CycleNb until);
  /// Adds a BB to the Graph
  void addBB(BB &bb);
  /// Gets the graph's BBs
  std::vector<BB> getBBs();
  /// Retrieves a list of edge IDs that are either incoming to or outgoing from
  /// a specified node
  std::vector<EdgeId> getInOutEdgesOfNode(const NodeId &nodeId);

private:
  /// Edges of the graph
  std::vector<GraphEdge> edges;
  /// Nodes of the graph mapped with their corresponding node identifier
  std::map<NodeId, GraphNode> nodes;
  /// State of each edge given a specific clock cycle
  CycleTransitions cycleEdgeStates;
  /// Map of the edges of the graph :
  /// ((src node id, outPort number), (dest node id, inPort number)) -> edge id
  std::map<std::pair<std::pair<NodeId, size_t>, std::pair<NodeId, size_t>>,
           EdgeId>
      mapEdges;
  /// BBs of the Graph
  std::vector<BB> bbs;
};

} // namespace visual
} // namespace dynamatic

#endif // DYNAMATIC_VISUAL_DATAFLOW_GRAPH_H
