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

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <queue>
#include <set>
#include <stack>

// Make the graph boost analyzable.
// NOTE: Moving this to the header file will cause linking errors.
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

void SynchronizingCyclesFinderGraph::buildFromCFDFC(
    handshake::FuncOp funcOp, const buffer::CFDFC &cfdfc) {
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

void SynchronizingCyclesFinderGraph::computeSccsAndBuildNonCyclicSubgraph() {
  size_t n = nodes.size();
  if (n == 0) return;

  // Kosaraju's algorithm for SCCs
  /// NOTE: Similar SCC code exists in experimental/Transforms/ResourceSharing/SharingSupport.cpp
  //  Kept separate because we need the non-cyclic adjacency list.

  // DFS to compute finishing order
  std::vector<bool> visited(n, false);
  std::stack<size_t> finishOrder;

  std::function<void(size_t)> dfs1 = [&](size_t u) {
    visited[u] = true;
    for (size_t edgeIdx : adjList[u]) {
      size_t v = edges[edgeIdx].dstId;
      if (!visited[v]) {
        dfs1(v);
      }
    }
    finishOrder.push(u);
  };

  for (size_t i = 0; i < n; ++i) {
    if (!visited[i]) {
      dfs1(i);
    }
  }

  // Build reverse adjacency list
  std::vector<std::vector<size_t>> revAdj(n);
  for (const auto &edge : edges) {
    revAdj[edge.dstId].push_back(edge.srcId);
  }

  // DFS on reverse graph in finish order to find SCCs
  nodeSccId.assign(n, 0);
  std::fill(visited.begin(), visited.end(), false);
  size_t sccCount = 0;

  std::function<void(size_t, size_t)> dfs2 = [&](size_t u, size_t sccId) {
    visited[u] = true;
    nodeSccId[u] = sccId;
    for (size_t v : revAdj[u]) {
      if (!visited[v]) {
        dfs2(v, sccId);
      }
    }
  };

  while (!finishOrder.empty()) {
    size_t u = finishOrder.top();
    finishOrder.pop();
    if (!visited[u]) {
      dfs2(u, sccCount);
      sccCount++;
    }
  }

  // Build non-cyclic adjacency list (only edges between different SCCs)
  nonCyclicAdjList.resize(n);
  for (size_t u = 0; u < n; ++u) {
    for (size_t edgeIdx : adjList[u]) {
      size_t v = edges[edgeIdx].dstId;
      // Only include edge if src and dst are in different SCCs
      if (nodeSccId[u] != nodeSccId[v]) {
        nonCyclicAdjList[u].push_back(edgeIdx);
      }
    }
  }

  LLVM_DEBUG(
    llvm::errs() << "Computed " << sccCount << " SCCs in CFDFC with " << n 
                 << " nodes. Non-cyclic subgraph has "
                 << std::accumulate(nonCyclicAdjList.begin(), nonCyclicAdjList.end(), 0UL,
                      [](size_t sum, const auto& v) { return sum + v.size(); })
                 << " edges.\n"
  );
}

std::vector<size_t> SynchronizingCyclesFinderGraph::getAllJoins() const {
  std::vector<size_t> joins;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (isJoinNode(i)) {
      joins.push_back(i);
    }
  }
  return joins;
}

std::vector<size_t> SynchronizingCyclesFinderGraph::findPathToJoin(
    const SimpleCycle &cycle, size_t joinId) const {
  
  std::set<size_t> cycleNodes(cycle.nodes.begin(), cycle.nodes.end());
  
  // BFS from cycle exit nodes
  std::map<size_t, size_t> parent;
  std::queue<size_t> bfs;

  // exit nodes are pretty much cycle nodes that are neighbors of nodes not in the cycle.
  for (size_t nodeId : cycleNodes) {
    for (size_t edgeIdx : nonCyclicAdjList[nodeId]) {
      size_t neighbor = edges[edgeIdx].dstId;
      if (!cycleNodes.count(neighbor) && !parent.count(neighbor)) {
        parent[neighbor] = nodeId;
        bfs.push(neighbor);
      }
    }
  }

  while (!bfs.empty()) {
    size_t current = bfs.front();
    bfs.pop();

    if (current == joinId) {
      // get path from cycle exit to join
      std::vector<size_t> path;
      size_t node = joinId;
      while (!cycleNodes.count(node)) {
        path.push_back(node);
        node = parent[node];
      }
      std::reverse(path.begin(), path.end());
      return path;
    }

    // continue BFS using non-cyclic edges
    for (size_t edgeIdx : nonCyclicAdjList[current]) {
      size_t neighbor = edges[edgeIdx].dstId;
      if (!cycleNodes.count(neighbor) && !parent.count(neighbor)) {
        parent[neighbor] = current;
        bfs.push(neighbor);
      }
    }
  }

  return {};
}

std::vector<SynchronizingCyclePair> 
SynchronizingCyclesFinderGraph::findSynchronizingCyclePairs() {
  computeSccsAndBuildNonCyclicSubgraph();

  std::vector<SimpleCycle> allCycles = findAllCycles();
  LLVM_DEBUG(
    llvm::errs() << "Found " << allCycles.size() << " cycles in CFDFC.\n"
  );

  std::vector<size_t> allJoins = getAllJoins();
  LLVM_DEBUG(
    llvm::errs() << "Found " << allJoins.size() << " joins in CFDFC.\n"
  );

  std::vector<SynchronizingCyclePair> pairs;

  for (size_t i = 0; i < allCycles.size(); ++i) {
    for (size_t j = i + 1; j < allCycles.size(); ++j) {
      const auto &cycleOne = allCycles[i];
      const auto &cycleTwo = allCycles[j];

      // Criteria 1: Cycles must be disjoint
      if (!cycleOne.isDisjointFrom(cycleTwo)) {
        continue;
      }

      // Criteria 2: Find joins reachable from BOTH cycles via non-cyclic paths
      std::vector<PathToJoin> pathsToJoins;

      for (size_t joinId : allJoins) {
        auto pathOne = findPathToJoin(cycleOne, joinId);
        auto pathTwo = findPathToJoin(cycleTwo, joinId);

        // Both cycles must be able to reach this join
        if (!pathOne.empty() && !pathTwo.empty()) {
          PathToJoin pathInfo(joinId);
          pathInfo.pathFromCycleOne = std::move(pathOne);
          pathInfo.pathFromCycleTwo = std::move(pathTwo);
          pathsToJoins.push_back(std::move(pathInfo));
        }
      }

      if (!pathsToJoins.empty()) {
        pairs.emplace_back(cycleOne, cycleTwo, std::move(pathsToJoins));
      }
    }
  }

  LLVM_DEBUG(
    llvm::errs() << "Found " << pairs.size() << " synchronizing cycle pairs.\n"
  );
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

/// GraphViz Plotting ///

static llvm::SmallString<256> resolveFullPath(llvm::StringRef filename) {
  llvm::SmallString<256> fullPath;
  if (llvm::sys::path::is_absolute(filename)) {
    fullPath = filename;
  } else {
    if (auto ec = llvm::sys::fs::current_path(fullPath)) {
      llvm::errs() << "Failed to get current path: " << ec.message() << "\n";
      fullPath = filename;
    } else {
      llvm::sys::path::append(fullPath, filename);
    }
  }
  return fullPath;
}

void SynchronizingCyclesFinderGraph::dumpSynchronizingCyclePair(
    const SynchronizingCyclePair &pair, llvm::StringRef filename) const {
  llvm::SmallString<256> fullPath = resolveFullPath(filename);

  std::ofstream file(fullPath.str().str());
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file: " << fullPath << "\n";
    return;
  }

  std::set<size_t> cycleOneNodes(pair.cycleOne.nodes.begin(),
                                  pair.cycleOne.nodes.end());
  std::set<size_t> cycleTwoNodes(pair.cycleTwo.nodes.begin(),
                                  pair.cycleTwo.nodes.end());

  // Collect all path nodes (excluding the join itself)
  std::set<size_t> pathOneNodes, pathTwoNodes;
  for (const auto &pathInfo : pair.pathsToJoins) {
    for (size_t i = 0; i + 1 < pathInfo.pathFromCycleOne.size(); ++i)
      pathOneNodes.insert(pathInfo.pathFromCycleOne[i]);
    for (size_t i = 0; i + 1 < pathInfo.pathFromCycleTwo.size(); ++i)
      pathTwoNodes.insert(pathInfo.pathFromCycleTwo[i]);
  }

  file << "digraph SynchronizingCyclePair {\n";
  file << "  rankdir=TB;\n";
  file << "  bgcolor=\"#fafafa\";\n";
  file << "  node [fontname=\"Helvetica\", fontsize=10];\n";
  file << "  edge [fontname=\"Helvetica\", fontsize=9];\n\n";

  // Cycle One cluster (green)
  file << "  subgraph cluster_cycle_one {\n";
  file << "    label=\"Cycle One (" << cycleOneNodes.size() << " nodes)\";\n";
  file << "    style=filled;\n";
  file << "    color=\"#2e7d32\";\n";
  file << "    fillcolor=\"#e8f5e9\";\n";
  file << "    fontcolor=\"#1b5e20\";\n";
  file << "    fontsize=12;\n\n";

  for (size_t nodeId : cycleOneNodes) {
    file << "    c1_" << getNodeDotId(nodeId) << " [label=\""
         << getNodeLabel(nodeId)
         << "\", style=filled, fillcolor=\"#a5d6a7\", color=\"#2e7d32\"];\n";
  }
  file << "\n";
  for (size_t i = 0; i < pair.cycleOne.nodes.size(); ++i) {
    size_t src = pair.cycleOne.nodes[i];
    size_t dst = pair.cycleOne.nodes[(i + 1) % pair.cycleOne.nodes.size()];
    file << "    c1_" << getNodeDotId(src) << " -> c1_" << getNodeDotId(dst)
         << " [color=\"#2e7d32\", penwidth=2];\n";
  }
  file << "  }\n\n";

  // Cycle Two cluster (blue)
  file << "  subgraph cluster_cycle_two {\n";
  file << "    label=\"Cycle Two (" << cycleTwoNodes.size() << " nodes)\";\n";
  file << "    style=filled;\n";
  file << "    color=\"#1565c0\";\n";
  file << "    fillcolor=\"#e3f2fd\";\n";
  file << "    fontcolor=\"#0d47a1\";\n";
  file << "    fontsize=12;\n\n";

  for (size_t nodeId : cycleTwoNodes) {
    file << "    c2_" << getNodeDotId(nodeId) << " [label=\""
         << getNodeLabel(nodeId)
         << "\", style=filled, fillcolor=\"#90caf9\", color=\"#1565c0\"];\n";
  }
  file << "\n";
  for (size_t i = 0; i < pair.cycleTwo.nodes.size(); ++i) {
    size_t src = pair.cycleTwo.nodes[i];
    size_t dst = pair.cycleTwo.nodes[(i + 1) % pair.cycleTwo.nodes.size()];
    file << "    c2_" << getNodeDotId(src) << " -> c2_" << getNodeDotId(dst)
         << " [color=\"#1565c0\", penwidth=2];\n";
  }
  file << "  }\n\n";

  // Path nodes from Cycle One (light green)
  if (!pathOneNodes.empty()) {
    file << "  // Path nodes from Cycle One to joins\n";
    for (size_t nodeId : pathOneNodes) {
      file << "  path1_" << getNodeDotId(nodeId) << " [label=\""
           << getNodeLabel(nodeId)
           << "\", style=filled, fillcolor=\"#c8e6c9\", color=\"#66bb6a\"];\n";
    }
    file << "\n";
  }

  // Path nodes from Cycle Two (light blue)
  if (!pathTwoNodes.empty()) {
    file << "  // Path nodes from Cycle Two to joins\n";
    for (size_t nodeId : pathTwoNodes) {
      file << "  path2_" << getNodeDotId(nodeId) << " [label=\""
           << getNodeLabel(nodeId)
           << "\", style=filled, fillcolor=\"#bbdefb\", color=\"#42a5f5\"];\n";
    }
    file << "\n";
  }

  // Common Joins cluster (orange)
  if (!pair.pathsToJoins.empty()) {
    file << "  subgraph cluster_common_joins {\n";
    file << "    label=\"Common Joins (" << pair.pathsToJoins.size() << ")\";\n";
    file << "    style=filled;\n";
    file << "    color=\"#e65100\";\n";
    file << "    fillcolor=\"#fff3e0\";\n";
    file << "    fontcolor=\"#bf360c\";\n";
    file << "    fontsize=12;\n\n";

    for (const auto &pathInfo : pair.pathsToJoins) {
      file << "    join_" << getNodeDotId(pathInfo.joinId) << " [label=\""
           << getNodeLabel(pathInfo.joinId)
           << "\", style=filled, fillcolor=\"#ffcc80\", color=\"#e65100\", penwidth=2];\n";
    }
    file << "  }\n\n";

    // Draw path edges (deduplicated)
    file << "  // Paths from cycles to joins (via non-cyclic subgraph)\n";
    std::set<std::pair<std::string, std::string>> drawnEdges;

    for (const auto &pathInfo : pair.pathsToJoins) {
      const auto &path1 = pathInfo.pathFromCycleOne;
      const auto &path2 = pathInfo.pathFromCycleTwo;

      // Path from Cycle One
      if (!path1.empty()) {
        // Find which cycle node connects to first path node
        for (size_t cycleNode : cycleOneNodes) {
          for (size_t edgeIdx : nonCyclicAdjList[cycleNode]) {
            if (edges[edgeIdx].dstId == path1[0]) {
              std::string src = "c1_" + getNodeDotId(cycleNode);
              std::string dst = (path1.size() == 1)
                  ? "join_" + getNodeDotId(path1[0])
                  : "path1_" + getNodeDotId(path1[0]);
              if (drawnEdges.insert({src, dst}).second) {
                file << "  " << src << " -> " << dst
                     << " [color=\"#2e7d32\", style=dashed, penwidth=1.5];\n";
              }
            }
          }
        }
        // Path edges
        for (size_t i = 0; i + 1 < path1.size(); ++i) {
          std::string src = "path1_" + getNodeDotId(path1[i]);
          std::string dst = (i + 2 == path1.size())
              ? "join_" + getNodeDotId(path1[i + 1])
              : "path1_" + getNodeDotId(path1[i + 1]);
          if (drawnEdges.insert({src, dst}).second) {
            file << "  " << src << " -> " << dst
                 << " [color=\"#2e7d32\", style=dashed, penwidth=1.5];\n";
          }
        }
      }

      // Path from Cycle Two
      if (!path2.empty()) {
        for (size_t cycleNode : cycleTwoNodes) {
          for (size_t edgeIdx : nonCyclicAdjList[cycleNode]) {
            if (edges[edgeIdx].dstId == path2[0]) {
              std::string src = "c2_" + getNodeDotId(cycleNode);
              std::string dst = (path2.size() == 1)
                  ? "join_" + getNodeDotId(path2[0])
                  : "path2_" + getNodeDotId(path2[0]);
              if (drawnEdges.insert({src, dst}).second) {
                file << "  " << src << " -> " << dst
                     << " [color=\"#1565c0\", style=dashed, penwidth=1.5];\n";
              }
            }
          }
        }
        for (size_t i = 0; i + 1 < path2.size(); ++i) {
          std::string src = "path2_" + getNodeDotId(path2[i]);
          std::string dst = (i + 2 == path2.size())
              ? "join_" + getNodeDotId(path2[i + 1])
              : "path2_" + getNodeDotId(path2[i + 1]);
          if (drawnEdges.insert({src, dst}).second) {
            file << "  " << src << " -> " << dst
                 << " [color=\"#1565c0\", style=dashed, penwidth=1.5];\n";
          }
        }
      }
    }
  }

  file << "}\n";
  file.close();
  llvm::errs() << "Dumped synchronizing cycle pair to " << fullPath << "\n";
}

void SynchronizingCyclesFinderGraph::dumpAllSynchronizingCyclePairs(
    const std::vector<SynchronizingCyclePair> &pairs,
    llvm::StringRef filename) const {
  llvm::SmallString<256> fullPath = resolveFullPath(filename);

  std::ofstream file(fullPath.str().str());
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file: " << fullPath << "\n";
    return;
  }

  file << "digraph AllSynchronizingCyclePairs {\n";
  file << "  rankdir=TB;\n";
  file << "  bgcolor=\"#fafafa\";\n";
  file << "  compound=true;\n";
  file << "  node [fontname=\"Helvetica\", fontsize=10];\n";
  file << "  edge [fontname=\"Helvetica\", fontsize=9];\n\n";

  for (size_t pairIdx = 0; pairIdx < pairs.size(); ++pairIdx) {
    const SynchronizingCyclePair &pair = pairs[pairIdx];
    std::string prefix = "p" + std::to_string(pairIdx) + "_";

    std::set<size_t> cycleOneNodes(pair.cycleOne.nodes.begin(),
                                    pair.cycleOne.nodes.end());
    std::set<size_t> cycleTwoNodes(pair.cycleTwo.nodes.begin(),
                                    pair.cycleTwo.nodes.end());

    // Collect path nodes
    std::set<size_t> pathOneNodes, pathTwoNodes;
    for (const auto &pathInfo : pair.pathsToJoins) {
      for (size_t i = 0; i + 1 < pathInfo.pathFromCycleOne.size(); ++i)
        pathOneNodes.insert(pathInfo.pathFromCycleOne[i]);
      for (size_t i = 0; i + 1 < pathInfo.pathFromCycleTwo.size(); ++i)
        pathTwoNodes.insert(pathInfo.pathFromCycleTwo[i]);
    }

    file << "  subgraph cluster_pair_" << pairIdx << " {\n";
    file << "    label=\"Pair " << pairIdx << " (" 
         << pair.pathsToJoins.size() << " common joins)\";\n";
    file << "    style=rounded;\n";
    file << "    color=\"#424242\";\n";
    file << "    bgcolor=\"#fafafa\";\n";
    file << "    fontsize=14;\n\n";

    // Cycle One
    file << "    subgraph cluster_" << prefix << "cycle_one {\n";
    file << "      label=\"Cycle One (" << cycleOneNodes.size() << " nodes)\";\n";
    file << "      style=filled;\n";
    file << "      color=\"#2e7d32\";\n";
    file << "      fillcolor=\"#e8f5e9\";\n";
    file << "      fontcolor=\"#1b5e20\";\n\n";

    for (size_t nodeId : cycleOneNodes) {
      file << "      " << prefix << "c1_" << getNodeDotId(nodeId)
           << " [label=\"" << getNodeLabel(nodeId)
           << "\", style=filled, fillcolor=\"#a5d6a7\", color=\"#2e7d32\"];\n";
    }
    file << "\n";
    for (size_t i = 0; i < pair.cycleOne.nodes.size(); ++i) {
      size_t src = pair.cycleOne.nodes[i];
      size_t dst = pair.cycleOne.nodes[(i + 1) % pair.cycleOne.nodes.size()];
      file << "      " << prefix << "c1_" << getNodeDotId(src) << " -> "
           << prefix << "c1_" << getNodeDotId(dst)
           << " [color=\"#2e7d32\", penwidth=2];\n";
    }
    file << "    }\n\n";

    // Cycle Two
    file << "    subgraph cluster_" << prefix << "cycle_two {\n";
    file << "      label=\"Cycle Two (" << cycleTwoNodes.size() << " nodes)\";\n";
    file << "      style=filled;\n";
    file << "      color=\"#1565c0\";\n";
    file << "      fillcolor=\"#e3f2fd\";\n";
    file << "      fontcolor=\"#0d47a1\";\n\n";

    for (size_t nodeId : cycleTwoNodes) {
      file << "      " << prefix << "c2_" << getNodeDotId(nodeId)
           << " [label=\"" << getNodeLabel(nodeId)
           << "\", style=filled, fillcolor=\"#90caf9\", color=\"#1565c0\"];\n";
    }
    file << "\n";
    for (size_t i = 0; i < pair.cycleTwo.nodes.size(); ++i) {
      size_t src = pair.cycleTwo.nodes[i];
      size_t dst = pair.cycleTwo.nodes[(i + 1) % pair.cycleTwo.nodes.size()];
      file << "      " << prefix << "c2_" << getNodeDotId(src) << " -> "
           << prefix << "c2_" << getNodeDotId(dst)
           << " [color=\"#1565c0\", penwidth=2];\n";
    }
    file << "    }\n\n";

    // Path nodes from Cycle One
    if (!pathOneNodes.empty()) {
      file << "    // Path nodes from Cycle One\n";
      for (size_t nodeId : pathOneNodes) {
        file << "    " << prefix << "path1_" << getNodeDotId(nodeId)
             << " [label=\"" << getNodeLabel(nodeId)
             << "\", style=filled, fillcolor=\"#c8e6c9\", color=\"#66bb6a\"];\n";
      }
      file << "\n";
    }

    // Path nodes from Cycle Two
    if (!pathTwoNodes.empty()) {
      file << "    // Path nodes from Cycle Two\n";
      for (size_t nodeId : pathTwoNodes) {
        file << "    " << prefix << "path2_" << getNodeDotId(nodeId)
             << " [label=\"" << getNodeLabel(nodeId)
             << "\", style=filled, fillcolor=\"#bbdefb\", color=\"#42a5f5\"];\n";
      }
      file << "\n";
    }

    // Common Joins
    if (!pair.pathsToJoins.empty()) {
      file << "    subgraph cluster_" << prefix << "joins {\n";
      file << "      label=\"Common Joins\";\n";
      file << "      style=filled;\n";
      file << "      color=\"#e65100\";\n";
      file << "      fillcolor=\"#fff3e0\";\n";
      file << "      fontcolor=\"#bf360c\";\n\n";

      for (const auto &pathInfo : pair.pathsToJoins) {
        file << "      " << prefix << "join_" << getNodeDotId(pathInfo.joinId)
             << " [label=\"" << getNodeLabel(pathInfo.joinId)
             << "\", style=filled, fillcolor=\"#ffcc80\", color=\"#e65100\", penwidth=2];\n";
      }
      file << "    }\n\n";

      // Draw path edges (deduplicated)
      std::set<std::pair<std::string, std::string>> drawnEdges;

      for (const auto &pathInfo : pair.pathsToJoins) {
        const auto &path1 = pathInfo.pathFromCycleOne;
        const auto &path2 = pathInfo.pathFromCycleTwo;

        // Path from Cycle One
        if (!path1.empty()) {
          for (size_t cycleNode : cycleOneNodes) {
            for (size_t edgeIdx : nonCyclicAdjList[cycleNode]) {
              if (edges[edgeIdx].dstId == path1[0]) {
                std::string src = prefix + "c1_" + getNodeDotId(cycleNode);
                std::string dst = (path1.size() == 1)
                    ? prefix + "join_" + getNodeDotId(path1[0])
                    : prefix + "path1_" + getNodeDotId(path1[0]);
                if (drawnEdges.insert({src, dst}).second) {
                  file << "    " << src << " -> " << dst
                       << " [color=\"#2e7d32\", style=dashed, penwidth=1.5];\n";
                }
              }
            }
          }
          for (size_t i = 0; i + 1 < path1.size(); ++i) {
            std::string src = prefix + "path1_" + getNodeDotId(path1[i]);
            std::string dst = (i + 2 == path1.size())
                ? prefix + "join_" + getNodeDotId(path1[i + 1])
                : prefix + "path1_" + getNodeDotId(path1[i + 1]);
            if (drawnEdges.insert({src, dst}).second) {
              file << "    " << src << " -> " << dst
                   << " [color=\"#2e7d32\", style=dashed, penwidth=1.5];\n";
            }
          }
        }

        // Path from Cycle Two
        if (!path2.empty()) {
          for (size_t cycleNode : cycleTwoNodes) {
            for (size_t edgeIdx : nonCyclicAdjList[cycleNode]) {
              if (edges[edgeIdx].dstId == path2[0]) {
                std::string src = prefix + "c2_" + getNodeDotId(cycleNode);
                std::string dst = (path2.size() == 1)
                    ? prefix + "join_" + getNodeDotId(path2[0])
                    : prefix + "path2_" + getNodeDotId(path2[0]);
                if (drawnEdges.insert({src, dst}).second) {
                  file << "    " << src << " -> " << dst
                       << " [color=\"#1565c0\", style=dashed, penwidth=1.5];\n";
                }
              }
            }
          }
          for (size_t i = 0; i + 1 < path2.size(); ++i) {
            std::string src = prefix + "path2_" + getNodeDotId(path2[i]);
            std::string dst = (i + 2 == path2.size())
                ? prefix + "join_" + getNodeDotId(path2[i + 1])
                : prefix + "path2_" + getNodeDotId(path2[i + 1]);
            if (drawnEdges.insert({src, dst}).second) {
              file << "    " << src << " -> " << dst
                   << " [color=\"#1565c0\", style=dashed, penwidth=1.5];\n";
            }
          }
        }
      }
    }

    file << "  }\n\n";
  }

  file << "}\n";
  file.close();
  llvm::errs() << "Dumped " << pairs.size() << " synchronizing cycle pairs to "
               << fullPath << "\n";
}

} // namespace dynamatic

