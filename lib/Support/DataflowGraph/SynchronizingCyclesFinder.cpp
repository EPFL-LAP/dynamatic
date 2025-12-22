//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class for finding pairs of synchronizing cycles in a dataflow graph.
// Given a dataflow graph, find pairs of cycles that are synchronizing, 
// according to the definition found in Section 4 of this paper:
//
// [Xu, Josipović, FPGA'24 (https://dl.acm.org/doi/10.1145/3626202.36375)] 
//
// Definition 3: Two cycles are a pair of synchronizing cycles in a
// dataflow circuit if the following properties hold: (1) The two cycles
// are disjoint (i.e. they do not have any common units) and belong to
// the same CFC (defined in Section 3.2). (2) There exists at least one
// join that is reachable from both cycles without crossing any edge
// on the cycle in the CFC they belong to.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/DataflowGraph/SynchronizingCyclesFinder.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"

#include <queue>

/// Make the graph boost analyzable.
/// NOTE: Moving this to the heade file will cause linking errors.

using BoostDataflowGraph = 
    boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>;

// See https://github.com/boostorg/graph/issues/182
namespace boost { 
  void renumber_vertex_indices(BoostDataflowGraph const&) {} 
} // namespace boost

namespace dynamatic {

/// SimpleCycle ///

bool SimpleCycle::isDisjointFrom(const SimpleCycle &other) const {
  std::set<size_t> thisNodes(nodes.begin(), nodes.end());
  return std::all_of(other.nodes.begin(), other.nodes.end(),
                   [&](size_t id) { return !thisNodes.count(id); });
}

/// SynchronizingCyclesGraph ///

size_t SynchronizingCyclesFinderGraph::getOrAddNode(mlir::Operation *op) {
  if (auto it = opToNodeId.find(op); it != opToNodeId.end())
    return it->second;
  size_t id = addNode(op);
  opToNodeId[op] = id;
  return id;
}

void SynchronizingCyclesFinderGraph::buildFromCFDFC(handshake::FuncOp funcOp, const buffer::CFDFC &cfdfc) {
  this->funcOp = funcOp;

  for (mlir::Operation *op : cfdfc.units) {
    getOrAddNode(op);
  }

    for (mlir::Value channel : cfdfc.channels) {
    mlir::Operation *producer = channel.getDefiningOp();
    if (!producer)
      continue;

    for (mlir::Operation *consumer : channel.getUsers()) {
      // Only add edge if both producer and consumer are in the CFDFC
      if (opToNodeId.count(producer) && opToNodeId.count(consumer)) {
        size_t srcId = opToNodeId[producer];
        size_t dstId = opToNodeId[consumer];
        addEdge(srcId, dstId, channel);
      }
    }
  }
}

bool SynchronizingCyclesFinderGraph::isForkNode(size_t nodeId) const {
  return isa<handshake::ForkOp>(nodes[nodeId].op) || 
        isa<handshake::LazyForkOp>(nodes[nodeId].op);
}

bool SynchronizingCyclesFinderGraph::isJoinNode(size_t nodeId) const {
  return isa<handshake::MuxOp>(nodes[nodeId].op) ||
        isa<handshake::ConditionalBranchOp>(nodes[nodeId].op);
}

std::string SynchronizingCyclesFinderGraph::getNodeLabel(size_t nodeId) const {
  return nodes[nodeId].op->getName().getStringRef().str();
}

std::string SynchronizingCyclesFinderGraph::getNodeDotId(size_t nodeId) const {
  return "node_" + std::to_string(nodeId);
}

std::set<size_t> SynchronizingCyclesFinderGraph::getReachableNodes(
    const std::set<size_t> &startNodes) const {
  std::set<size_t> reachable;
  std::queue<size_t> bfs;

  for (size_t start : startNodes) {
    bfs.push(start);
    reachable.insert(start);
  }

  while (!bfs.empty()) {
    size_t current = bfs.front();
    bfs.pop();

    for (size_t edgeIdx : adjList[current]) {
      size_t neighbor = edges[edgeIdx].dstId;
      if (!reachable.count(neighbor)) {
        reachable.insert(neighbor);
        bfs.push(neighbor);
      }
    }
  }
  return reachable;
}

std::set<size_t> SynchronizingCyclesFinderGraph::findReachableJoins(
  const SimpleCycle &cycle) const {

  std::set<size_t> cycleNodes(cycle.nodes.begin(), cycle.nodes.end());

  // Find all successors of cycle nodes that are not in the cycle
  std::set<size_t> exitNodes;
  for (size_t nodeId : cycleNodes) {
    for (size_t edgeIdx : adjList[nodeId]) {
      size_t neighbor = edges[edgeIdx].dstId;
      if (!cycleNodes.count(neighbor)) {
        exitNodes.insert(neighbor);
      }
    }
  }

  // BFS from exit nodes to find all reachable joins
  std::set<size_t> reachable = getReachableNodes(exitNodes);
  std::set<size_t> reachableJoins;
  for (size_t nodeId : reachable) {
    if (isJoinNode(nodeId)) {
      reachableJoins.insert(nodeId);
    }
  }
  return reachableJoins;
}

std::vector<SynchronizingCyclePair> SynchronizingCyclesFinderGraph::findSynchronizingCyclePairs() const {
  std::vector<SimpleCycle> allCycles = findAllCycles();
  std::vector<SynchronizingCyclePair> pairs;

  std::vector<std::set<size_t>> reachableJoinsPerCycle;
  reachableJoinsPerCycle.reserve(allCycles.size());

  for (const SimpleCycle &cycle : allCycles) {
    reachableJoinsPerCycle.push_back(findReachableJoins(cycle));
  }

  for (size_t i = 0; i < allCycles.size(); ++i) {
    for (size_t j = i + 1; j < allCycles.size(); ++j) {
      const auto &one = allCycles[i];
      const auto &two = allCycles[j];

      // 1st Criteria: Cycles must be disjoint
      if (!one.isDisjointFrom(two)) {
        continue;
      }

      // 3rd Criteria: Must share at least one reachable join.
      std::set<size_t> commonJoins;
      std::set_intersection(
          reachableJoinsPerCycle[i].begin(), reachableJoinsPerCycle[i].end(),
          reachableJoinsPerCycle[j].begin(), reachableJoinsPerCycle[j].end(),
          std::inserter(commonJoins, commonJoins.begin()));

      if (!commonJoins.empty()) {
        pairs.emplace_back(one, two, std::move(commonJoins));
      }
    }
  }

  return pairs;
}

std::vector<SimpleCycle> SynchronizingCyclesFinderGraph::findAllCycles() const {
  // Build a Boost graph from our adjacency list
  BoostDataflowGraph boostGraph(nodes.size());
  
  for (const auto &edge : edges) {
    boost::add_edge(edge.srcId, edge.dstId, boostGraph);
  }

  std::vector<SimpleCycle> allCycles;
  CycleCollector collector{allCycles};
  
  boost::tiernan_all_cycles(boostGraph, collector);
  
  return allCycles;
}

} // namespace dynamatic