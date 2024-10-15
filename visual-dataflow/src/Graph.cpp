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
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace mlir;
using namespace dynamatic::visual;

void Graph::addEdge(GraphEdge edge) {
  edges.push_back(edge);
  NodePortPair srcPort =
      std::pair(edge.getSrcNode().getNodeId(), edge.getOutPort());
  NodePortPair dstPort =
      std::pair(edge.getDstNode().getNodeId(), edge.getInPort());
  EdgePorts edgePorts(srcPort, dstPort);
  mapEdges.insert(std::pair(edgePorts, edge.getEdgeId()));
}

void Graph::addNode(GraphNode node) { nodes.insert({node.getNodeId(), node}); }

LogicalResult Graph::getNode(NodeId &id, GraphNode &result) {
  if (auto nodeIt = nodes.find(id); nodeIt != nodes.end()) {
    result = nodeIt->second;
    return success();
  }
  return failure();
}

LogicalResult Graph::getEdgeId(EdgePorts &edgeInfo, EdgeId &edgeId) {
  if (auto edgeIt = mapEdges.find(edgeInfo); edgeIt != mapEdges.end()) {
    edgeId = edgeIt->second;
    return success();
  }
  return failure();
}

void Graph::addEdgeState(CycleNb cycleNb, EdgeId edgeId, State state,
                         const Data &data) {
  if (cycleNb == 0) {
    ChannelTransitions mapEdgeState = cycleEdgeStates[0];
    std::pair<State, Data> statePair = std::pair(state, data);
    mapEdgeState.insert(std::pair(edgeId, statePair));
    cycleEdgeStates[cycleNb] = mapEdgeState;
  } else {
    std::pair<State, Data> statePair = std::pair(state, data);
    cycleEdgeStates[cycleNb].at(edgeId) = statePair;
  }
}

CycleTransitions &Graph::getCycleEdgeStates() { return cycleEdgeStates; }

std::map<NodeId, GraphNode> Graph::getNodes() { return nodes; }

std::vector<GraphEdge> Graph::getEdges() { return edges; }

void Graph::dupilcateEdgeStates(CycleNb from, CycleNb until) {
  const ChannelTransitions &initialMapEdgeState = cycleEdgeStates[from];
  for (CycleNb i = from + 1; i <= until; i++)
    cycleEdgeStates[i] = ChannelTransitions{initialMapEdgeState};
}

void Graph::addBB(BB &bb) { this->bbs.push_back(bb); }

std::vector<BB> Graph::getBBs() { return bbs; }

std::vector<EdgeId> Graph::getInOutEdgesOfNode(const NodeId &nodeId) {
  std::vector<EdgeId> edgeList;
  for (auto &edge : edges) {
    if (edge.getSrcNode().getNodeId() == nodeId ||
        edge.getDstNode().getNodeId() == nodeId)
      edgeList.push_back(edge.getEdgeId());
  }
  return edgeList;
}
