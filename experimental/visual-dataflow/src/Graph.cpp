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
#include <utility>

using namespace mlir;
using namespace dynamatic::experimental::visual_dataflow;

void Graph::addEdge(GraphEdge edge) {
  edges.push_back(edge);
  std::pair<NodeId, size_t> srcInfo =
      std::pair(edge.getSrcNode().getNodeId(), edge.getOutPort());
  std::pair<NodeId, size_t> dstInfo =
      std::pair(edge.getDstNode().getNodeId(), edge.getInPort());
  std::pair<std::pair<NodeId, size_t>, std::pair<NodeId, size_t>> edgeInfo =
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
    std::pair<std::pair<NodeId, size_t>, std::pair<NodeId, size_t>> &edgeInfo,
    EdgeId &edgeId) {
  if (mapEdges.count(edgeInfo)) {
    edgeId = mapEdges.at(edgeInfo);
    return success();
  }
  return failure();
}

void Graph::addEdgeState(CycleNb cycleNb, EdgeId edgeId, State state,
                         Data data) {
  if (cycleNb == 0) {
    std::map<EdgeId, std::pair<State, Data>> mapEdgeState = cycleEdgeStates[0];
    std::pair<State, Data> statePair = std::pair(state, data);
    mapEdgeState.insert(std::pair(edgeId, statePair));
    cycleEdgeStates[cycleNb] = mapEdgeState;
  } else {
    std::pair<State, Data> statePair = std::pair(state, data);
    cycleEdgeStates[cycleNb].at(edgeId) = statePair;
  }
}

std::map<CycleNb, std::map<EdgeId, std::pair<State, Data>>>
Graph::getCycleEdgeStates() {
  return cycleEdgeStates;
}

std::map<NodeId, GraphNode> Graph::getNodes() { return nodes; }

std::vector<GraphEdge> Graph::getEdges() { return edges; }

void Graph::dupilcateEdgeStates(CycleNb from, CycleNb until) {
  const std::map<EdgeId, std::pair<State, Data>> &initialMapEdgeState =
      cycleEdgeStates[from];

  for (CycleNb i = from + 1; i <= until; i++) {
    std::map<EdgeId, std::pair<State, Data>> newMapEdgeState;

    for (const auto &pair : initialMapEdgeState) {
      newMapEdgeState[pair.first] = pair.second;
    }

    cycleEdgeStates[i] = newMapEdgeState;
  }
}

void Graph::addBB(BB bb) { this->bbs.push_back(bb); }

std::vector<BB> Graph::getBBs() { return bbs; }
