//===- Graph.cpp - Represents a graph ---------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a Graph.
//
//===----------------------------------------------------------------------===//
#include "Graph.h"
#include <functional>
#include <iostream>

using namespace mlir;
using namespace dynamatic::experimental::visual_dataflow;

void Graph::addEdge(GraphEdge edge) {
  edges.push_back(edge);
  std::pair<NodeId, int> srcInfo =
      std::pair(edge.getSrcNode().getNodeId(), edge.getOutPort());
  std::pair<NodeId, int> dstInfo =
      std::pair(edge.getDstNode().getNodeId(), edge.getInPort());
  std::pair<std::pair<NodeId, int>, std::pair<NodeId, int>> edgeInfo =
      std::pair(srcInfo, dstInfo);
  mapEdges.insert(std::pair(edgeInfo, edge.getEdgeId()));
}

void Graph::addNode(GraphNode node) { nodes.insert({node.getNodeId(), node}); }

LogicalResult Graph::getNode(NodeId &id, GraphNode &result) {
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

std::map<CycleNb, std::map<EdgeId, State>> Graph::getCycleEdgeStates() {
  return cycleEdgeStates;
}
std::map<NodeId, GraphNode> Graph::getNodes() { return nodes; }

std::vector<GraphEdge> Graph::getEdges() { return edges; }
