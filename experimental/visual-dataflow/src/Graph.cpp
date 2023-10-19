//===- Graph.cpp - Represents a graph ------------*- C++ -*-===//
//
// This file contains the implementation of a Graph.
//
//===----------------------------------------------------------------------===//
#include "Graph.h"
#include "GraphNode.h"
#include "mlir/Support/LogicalResult.h"
#include <utility>

using namespace dynamatic::experimental::visual_dataflow;

Graph::Graph(GraphId id) : id(id) {
  edges = std::vector<GraphEdge *>();
  nodes = std::map<std::string, GraphNode *>();
  cycleEdgeStates = std::map<CycleNb, std::map<EdgeId, State>>();
  mapEdges = std::map<std::pair<std::pair<NodeId, int>, std::pair<NodeId, int>>,
                      EdgeId>();
}

void Graph::addEdge(GraphEdge *edge) {
  edges.push_back(edge);
  std::pair<NodeId, int> srcInfo =
      std::pair(edge->getSrcNode()->getNodeId(), edge->getOutPort());
  std::pair<NodeId, int> dstInfo =
      std::pair(edge->getDstNode()->getNodeId(), edge->getInPort());
  std::pair<std::pair<NodeId, int>, std::pair<NodeId, int>> edgeInfo =
      std::pair(srcInfo, dstInfo);
  mapEdges.insert(std::pair(edgeInfo, edge->getEdgeId()));
}

void Graph::addNode(GraphNode *node) {
  nodes.insert({node->getNodeId(), node});
}

LogicalResult Graph::getNode(NodeId &id, GraphNode *&result) {
  if (nodes.count(id)) {
    result = nodes.at(id);
    return success();
  }
  return failure();
}

LogicalResult Graph::getEdgeId(
    std::pair<std::pair<NodeId, int>, std::pair<NodeId, int>> &edgeInfo,
    EdgeId &edgeId) {
  if (mapEdges.count(edgeInfo)) {
    edgeId = mapEdges.at(edgeInfo);
    return success();
  }
  return failure();
}

void Graph::addEdgeState(CycleNb cycleNb, EdgeId edgeId, State state) {
  // Creates an empty map if the key cycleNb is not found in the current
  // cycleEdgeStates map
  std::map<EdgeId, State> mapEdgeState = cycleEdgeStates[cycleNb];
  mapEdgeState.insert(std::pair(edgeId, state));
  cycleEdgeStates[cycleNb] = mapEdgeState;
}