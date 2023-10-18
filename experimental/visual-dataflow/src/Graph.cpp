//===- Graph.cpp - Represents a graph ------------*- C++ -*-===//
//
// This file contains the implementation of a Graph.
//
//===----------------------------------------------------------------------===//
#include "Graph.h"
#include <utility>

using namespace dynamatic::experimental::visual_dataflow;

Graph::Graph(GraphId id) : id(id) {
  edges = std::vector<GraphEdge *>();
  nodes = std::map<std::string, GraphNode *>();
  cycleEdgeStates = std::map<CycleNb, std::map<EdgeId, State>>();
}

void Graph::addEdge(GraphEdge *edge) { edges.push_back(edge); }

void Graph::addNode(GraphNode *node) {
  nodes.insert({node->getNodeId(), node});
}

void Graph::addCycleStates(CycleNb cycleNb,
                           std::map<EdgeId, State> mapEdgeState) {
  cycleEdgeStates[cycleNb] = std::move(mapEdgeState);
}

GraphNode *Graph::getNode(NodeId &id) { return nodes.at(id); }