//===- Graph.h - Represents a graph ------------*- C++ -*-===//
//
// The Graph class represents a graph composed of nodes and edges with the
// purpose of being displayed in Godot.
//===----------------------------------------------------------------------===//

#ifndef VISUAL_DATAFLOW_GRAPH_H
#define VISUAL_DATAFLOW_GRAPH_H

#include "GraphEdge.h"
#include "GraphNode.h"
#include <map>
#include <string>
#include <vector>

using CycleNb = int;
using GraphId = int;

/// State of an edge of the graph
enum State { UNDEFINED, READY, EMPTY, VALID, VALID_READY };

/// Implements the logic to create and update a Graph
class Graph {

public:
  /// Constructs a graph
  Graph(GraphId id);
  /// Adds an edge to the graph
  void addEdge(GraphEdge *edge);
  /// Adds a node to the graph
  void addNode(GraphNode *node);
  /// Adds the state of each edge for a given clock cycle
  void addCycleStates(CycleNb cycleNb, std::map<EdgeId, State> mapEdgeState);
  /// Retrieves a node based on a giver node identifier
  GraphNode *getNode(NodeId id);

private:
  /// Graph identifier
  GraphId id;
  /// Edges of the graph
  std::vector<GraphEdge *> edges;
  /// Nodes of the graph mapped with their corresponding node identifier
  std::map<NodeId, GraphNode *> nodes;
  /// State of each edge given a specific clock cycle
  std::map<CycleNb, std::map<EdgeId, State>> cycleEdgeStates;
};

#endif // VISUAL_DATAFLOW_GRAPH_H