//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for latency and occupancy balancing support.
// Implementation is based on the following paper:
// [Xu, Josipović, FPGA'24 (https://dl.acm.org/doi/10.1145/3626202.36375)]
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/LatencyAndOccupancyBalancingSupport.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <fstream>
#include <queue>

// Make the graph boost analyzable.
// NOTE: Moving this to the header file will cause linking errors.
using BoostDataflowSubgraph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>;

// See https://github.com/boostorg/graph/issues/182
namespace boost {
void renumber_vertex_indices(BoostDataflowSubgraph const &) {}
} // namespace boost

namespace dynamatic {

///=== RECONVERGENT PATH FINDER ===///

std::string ReconvergentPathFinderGraph::getNodeLabel(NodeIdType nodeId) const {
  std::string opName = nodes[nodeId].op->getName().getStringRef().str();
  return opName + "\\nStep: " + std::to_string(getNodeStep(nodeId));
}

std::string ReconvergentPathFinderGraph::getNodeDotId(NodeIdType nodeId) const {
  return "node_" + std::to_string(nodeId);
}

void ReconvergentPathFinderGraph::buildGraphFromSequence(
    handshake::FuncOp funcOp, llvm::ArrayRef<ArchBB> sequence) {
  if (sequence.empty()) {
    return;
  }

  this->nodes.clear();
  this->edges.clear();
  this->adjList.clear();
  this->revAdjList.clear();
  this->nodeIdToStep.clear();

  this->funcOp = funcOp;
  LogicBBs logicBBs = getLogicBBs(funcOp);

  // Map each step to its BB
  // =======================

  // Step 0 is the first srcBB, then each transition adds its dstBB as the next
  // step.
  stepToBB[0] = sequence[0].srcBB;
  for (size_t i = 0; i < sequence.size(); ++i) {
    stepToBB[i + 1] = sequence[i].dstBB;
  }

  // Populate nodes in order of steps.
  // =================================

  for (const auto &[step, bb] : stepToBB) {
    if (logicBBs.blocks.count(bb)) {
      for (Operation *op : logicBBs.blocks[bb]) {
        getOrAddNode(op, step);
      }
    }
  }

  // Populate intra-BB edges (within the same step).
  // ===============================================
  // Use nodeMap for O(1) lookup instead of linear search.
  // IMPORTANT: Skip backedges - they cross to the next iteration, not same
  // step.

  for (const auto &node : nodes) {
    unsigned step = getNodeStep(node.id);
    Operation *op = node.op;

    for (Value result : op->getResults()) {
      // Backedges go to the next iteration (inter-BB), not within same step
      if (isBackedge(result))
        continue;

      for (Operation *user : result.getUsers()) {
        // Check if user exists at the same step
        if (nodeMap.count({user, step})) {
          addEdge(node.id, nodeMap[{user, step}], result,
                  DataflowGraphEdgeType::INTRA_BB);
        }
      }
    }
  }

  // Populate inter-BB edges (between consecutive steps along the sequence).
  // =======================================================================
  // For transition i: srcStep = i, dstStep = i+1.
  // For self-loops (srcBB == dstBB), only backedge channels cross iterations.

  for (size_t i = 0; i < sequence.size(); ++i) {
    unsigned srcStep = i;
    unsigned dstStep = i + 1;
    bool isSelfLoop = (sequence[i].srcBB == sequence[i].dstBB);

    for (const auto &node : nodes) {
      if (getNodeStep(node.id) != srcStep)
        continue;

      Operation *op = node.op;
      for (Value result : op->getResults()) {
        // For self-loops, only backedge channels should create inter-BB edges.
        // Regular forward edges within the BB stay as intra-BB edges.
        if (isSelfLoop && !isBackedge(result))
          continue;

        for (Operation *user : result.getUsers()) {
          // Check if user exists at the destination step of this arch
          if (nodeMap.count({user, dstStep})) {
            addEdge(node.id, nodeMap[{user, dstStep}], result,
                    DataflowGraphEdgeType::INTER_BB);
          }
        }
      }
    }
  }
}

std::vector<ReconvergentPath>
ReconvergentPathFinderGraph::findReconvergentPaths() const {
  std::vector<NodeIdType> forks;
  std::vector<NodeIdType> joins;

  for (NodeIdType i = 0; i < nodes.size(); ++i) {
    if (isForkNode(i)) {
      forks.push_back(i);
    } else if (isJoinNode(i)) {
      joins.push_back(i);
    }
  }

  // Track unique node sets we've seen
  std::set<std::set<NodeIdType>> seenNodeSets;
  std::vector<ReconvergentPath> paths;

  for (NodeIdType forkId : forks) {
    // BFS forward from fork to find all reachable nodes
    std::vector<bool> reachableFromFork(nodes.size(), false);
    std::queue<NodeIdType> fwdQueue;
    fwdQueue.push(forkId);
    reachableFromFork[forkId] = true;

    while (!fwdQueue.empty()) {
      NodeIdType u = fwdQueue.front();
      fwdQueue.pop();
      for (size_t edgeId : adjList[u]) {
        NodeIdType v = edges[edgeId].dstId;
        if (!reachableFromFork[v]) {
          reachableFromFork[v] = true;
          fwdQueue.push(v);
        }
      }
    }

    for (NodeIdType joinId : joins) {
      // Skip if join is not reachable from fork
      if (!reachableFromFork[joinId])
        continue;

      // BFS backward from join to find all nodes that can reach it
      std::vector<bool> canReachJoin(nodes.size(), false);
      std::queue<NodeIdType> bwdQueue;
      bwdQueue.push(joinId);
      canReachJoin[joinId] = true;

      while (!bwdQueue.empty()) {
        NodeIdType u = bwdQueue.front();
        bwdQueue.pop();
        for (size_t edgeId : revAdjList[u]) {
          NodeIdType v = edges[edgeId].srcId;
          if (!canReachJoin[v]) {
            canReachJoin[v] = true;
            bwdQueue.push(v);
          }
        }
      }

      // The fork must have >=2 direct successors that can reach the join.
      // Otherwise it's just a linear chain, not actual
      // divergence/reconvergence.
      unsigned numDivergingPaths = 0;
      for (size_t edgeId : adjList[forkId]) {
        NodeIdType successor = edges[edgeId].dstId;
        if (canReachJoin[successor])
          numDivergingPaths++;
      }

      // Skip if only 0 or 1 path from fork reaches join (not reconvergent)
      if (numDivergingPaths < 2)
        continue;

      // Nodes reachable from fork AND can reach join
      std::set<NodeIdType> intersection;
      for (NodeIdType i = 0; i < nodes.size(); ++i) {
        if (reachableFromFork[i] && canReachJoin[i])
          intersection.insert(i);
      }

      // Skip trivial paths and deduplicate identical node sets
      if (intersection.size() > 2 && !seenNodeSets.count(intersection)) {
        seenNodeSets.insert(intersection);
        paths.emplace_back(forkId, joinId, std::move(intersection));
      }
    }
  }

  LLVM_DEBUG(llvm::errs() << "Found " << paths.size()
                          << " reconvergent paths from " << forks.size()
                          << " forks and " << joins.size() << " joins.\n";);

  return paths;
}

// [START AI-generated code]

void ReconvergentPathFinderGraph::dumpReconvergentPaths(
    llvm::ArrayRef<ReconvergentPath> paths, llvm::StringRef filename) const {
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

  // I don't understand why .str().str() but hey it fixes the issue.
  std::ofstream file(fullPath.str().str());
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file: " << fullPath << "\n";
    return;
  }

  file << "digraph ReconvergentPaths {\n";
  file << "  rankdir=TB;\n";
  file << "  bgcolor=white;\n";
  file << "  compound=true;\n\n";

  for (size_t pathIdx = 0; pathIdx < paths.size(); ++pathIdx) {
    const ReconvergentPath &path = paths[pathIdx];

    file << "  subgraph cluster_path_" << pathIdx << " {\n";
    file << "    label=\"Path " << pathIdx
         << " (Fork: " << getNodeLabel(path.forkNodeId)
         << " -> Join: " << getNodeLabel(path.joinNodeId) << ")\";\n";
    file << "    style=rounded;\n";
    file << "    color=blue;\n";
    file << "    bgcolor=\"#e8f4fc\";\n\n";

    // Emit nodes with unique IDs per path to avoid conflicts
    for (NodeIdType nodeId : path.nodeIds) {
      std::string uniqueId =
          "p" + std::to_string(pathIdx) + "_" + getNodeDotId(nodeId);
      std::string color = "";
      if (nodeId == path.forkNodeId)
        color = ", style=filled, fillcolor=\"#90EE90\""; // Green for fork
      else if (nodeId == path.joinNodeId)
        color = ", style=filled, fillcolor=\"#FFB6C1\""; // Pink for join

      file << "    " << uniqueId << " [label=\"" << getNodeLabel(nodeId) << "\""
           << color << "];\n";
    }

    file << "\n";

    // Emit edges within this path (different styles for intra/inter-BB)
    for (NodeIdType srcId : path.nodeIds) {
      for (size_t edgeId : adjList[srcId]) {
        auto &edge = edges[edgeId];
        NodeIdType dstId = edge.dstId;
        if (path.nodeIds.count(dstId)) {
          std::string srcUniqueId =
              "p" + std::to_string(pathIdx) + "_" + getNodeDotId(srcId);
          std::string dstUniqueId =
              "p" + std::to_string(pathIdx) + "_" + getNodeDotId(dstId);
          std::string style = (edge.type == DataflowGraphEdgeType::INTRA_BB)
                                  ? "solid"
                                  : "dashed";
          std::string color =
              (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "black" : "blue";
          file << "    " << srcUniqueId << " -> " << dstUniqueId
               << " [style=" << style << ", color=" << color << "];\n";
        }
      }
    }

    file << "  }\n\n";
  }

  file << "}\n";
  file.close();
  LLVM_DEBUG(llvm::errs() << "Dumped " << paths.size()
                          << " reconvergent paths to " << fullPath << "\n";);
}

void ReconvergentPathFinderGraph::dumpTransitionGraph(
    llvm::StringRef filename) const {
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

  std::ofstream file(fullPath.str().str());
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file: " << fullPath << "\n";
    return;
  }

  file << "digraph DataflowGraph {\n";
  file << "  rankdir=TB;\n";
  file << "  bgcolor=white;\n";
  file << "  compound=true;\n\n";

  // Group nodes by step to create clusters
  std::map<unsigned, std::vector<const DataflowGraphNode *>> nodesByStep;
  for (const auto &node : nodes) {
    nodesByStep[getNodeStep(node.id)].push_back(&node);
  }

  // Emit nodes grouped by step in subgraph clusters
  for (const auto &[step, stepNodes] : nodesByStep) {
    unsigned bbID = stepToBB.count(step) ? stepToBB.at(step) : 999;

    file << "  subgraph cluster_step_" << step << " {\n";
    file << "    label=\"Step " << step << " (BB " << bbID << ")\";\n";
    file << "    style=solid;\n";
    file << "    color=black;\n";
    file << "    bgcolor=\"#f0f0f0\";\n";

    for (const auto *node : stepNodes) {
      file << "    " << getNodeDotId(node->id) << " [label=\""
           << getNodeLabel(node->id) << "\"];\n";
    }
    file << "  }\n\n";
  }

  // Emit edges with different styles
  for (const auto &edge : edges) {
    std::string style =
        (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "solid" : "dashed";
    std::string color =
        (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "black" : "blue";

    file << "  " << getNodeDotId(edge.srcId) << " -> "
         << getNodeDotId(edge.dstId) << " [style=" << style
         << ", color=" << color << "];\n";
  }

  file << "}\n";
  file.close();
  LLVM_DEBUG(llvm::errs() << "Dumped DataflowGraph to " << fullPath << "\n";);
}

void ReconvergentPathFinderGraph::dumpAllGraphs(
    llvm::ArrayRef<ReconvergentPathFinderGraph> graphs,
    llvm::StringRef filename) {
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

  std::ofstream file(fullPath.str().str());
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file: " << fullPath << "\n";
    return;
  }

  file << "digraph AllDataflowGraphs {\n";
  file << "  rankdir=TB;\n";
  file << "  bgcolor=white;\n";
  file << "  compound=true;\n\n";

  for (size_t graphIdx = 0; graphIdx < graphs.size(); ++graphIdx) {
    const ReconvergentPathFinderGraph &graph = graphs[graphIdx];
    std::string graphPrefix = "g" + std::to_string(graphIdx) + "_";

    file << "  subgraph cluster_graph_" << graphIdx << " {\n";
    file << "    label=\"Sequence " << graphIdx << "\";\n";
    file << "    style=bold;\n";
    file << "    color=darkblue;\n";
    file << "    bgcolor=\"#f8f8ff\";\n\n";

    // Group nodes by step to create nested clusters
    std::map<unsigned, std::vector<NodeIdType>> nodesByStep;
    for (const auto &node : graph.nodes) {
      nodesByStep[graph.getNodeStep(node.id)].push_back(node.id);
    }

    // Emit nodes grouped by step in subgraph clusters
    for (const auto &[step, stepNodeIds] : nodesByStep) {
      unsigned bbID = graph.getStepBB(step);
      file << "    subgraph cluster_" << graphPrefix << "step_" << step
           << " {\n";
      file << "      label=\"Step " << step << " (BB " << bbID << ")\";\n";
      file << "      style=solid;\n";
      file << "      color=black;\n";
      file << "      bgcolor=\"#f0f0f0\";\n";

      for (NodeIdType nodeId : stepNodeIds) {
        file << "      " << graphPrefix << graph.getNodeDotId(nodeId)
             << " [label=\"" << graph.getNodeLabel(nodeId) << "\"];\n";
      }
      file << "    }\n\n";
    }

    // Emit edges with different styles
    for (const auto &edge : graph.edges) {
      std::string style =
          (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "solid" : "dashed";
      std::string color =
          (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "black" : "blue";

      file << "    " << graphPrefix << graph.getNodeDotId(edge.srcId) << " -> "
           << graphPrefix << graph.getNodeDotId(edge.dstId)
           << " [style=" << style << ", color=" << color << "];\n";
    }

    file << "  }\n\n";
  }

  file << "}\n";
  file.close();
  LLVM_DEBUG(llvm::errs() << "Dumped " << graphs.size()
                          << " dataflow graphs to " << fullPath << "\n";);
}

void ReconvergentPathFinderGraph::dumpAllReconvergentPaths(
    llvm::ArrayRef<GraphPathsForDumping> graphPaths, llvm::StringRef filename) {
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

  std::ofstream file(fullPath.str().str());
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file: " << fullPath << "\n";
    return;
  }

  file << "digraph AllReconvergentPaths {\n";
  file << "  rankdir=TB;\n";
  file << "  bgcolor=white;\n";
  file << "  compound=true;\n\n";

  for (const auto &entry : graphPaths) {
    const ReconvergentPathFinderGraph *graph = entry.graph;
    const std::vector<ReconvergentPath> &paths = entry.paths;
    size_t graphIdx = entry.graphIndex;

    for (size_t pathIdx = 0; pathIdx < paths.size(); ++pathIdx) {
      const ReconvergentPath &path = paths[pathIdx];
      std::string uniquePrefix =
          "g" + std::to_string(graphIdx) + "_p" + std::to_string(pathIdx) + "_";

      file << "  subgraph cluster_" << uniquePrefix << " {\n";
      file << "    label=\"Seq " << graphIdx << " / Path " << pathIdx
           << " (Fork: " << graph->getNodeLabel(path.forkNodeId)
           << " -> Join: " << graph->getNodeLabel(path.joinNodeId) << ")\";\n";
      file << "    style=rounded;\n";
      file << "    color=blue;\n";
      file << "    bgcolor=\"#e8f4fc\";\n\n";

      // Emit nodes with unique IDs
      for (NodeIdType nodeId : path.nodeIds) {
        std::string uniqueId = uniquePrefix + graph->getNodeDotId(nodeId);
        std::string color = "";
        if (nodeId == path.forkNodeId)
          color = ", style=filled, fillcolor=\"#90EE90\""; // Green for fork
        else if (nodeId == path.joinNodeId)
          color = ", style=filled, fillcolor=\"#FFB6C1\""; // Pink for join

        file << "    " << uniqueId << " [label=\""
             << graph->getNodeLabel(nodeId) << "\"" << color << "];\n";
      }

      file << "\n";

      // Emit edges within this path
      for (NodeIdType srcId : path.nodeIds) {
        for (size_t edgeId : graph->adjList[srcId]) {
          const auto &edge = graph->edges[edgeId];
          NodeIdType dstId = edge.dstId;
          if (path.nodeIds.count(dstId)) {
            std::string srcUniqueId = uniquePrefix + graph->getNodeDotId(srcId);
            std::string dstUniqueId = uniquePrefix + graph->getNodeDotId(dstId);
            std::string style = (edge.type == DataflowGraphEdgeType::INTRA_BB)
                                    ? "solid"
                                    : "dashed";
            std::string edgeColor =
                (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "black"
                                                               : "blue";
            file << "    " << srcUniqueId << " -> " << dstUniqueId
                 << " [style=" << style << ", color=" << edgeColor << "];\n";
          }
        }
      }

      file << "  }\n\n";
    }
  }

  file << "}\n";
  file.close();
  LLVM_DEBUG(llvm::errs() << "Dumped " << totalPaths
                          << " reconvergent paths from " << graphPaths.size()
                          << " graphs to " << fullPath << "\n";);
}

// [END AI-generated code]

bool SimpleCycle::isDisjointFrom(const SimpleCycle &other) const {
  std::set<NodeIdType> thisNodes(nodes.begin(), nodes.end());
  return std::all_of(other.nodes.begin(), other.nodes.end(),
                     [&](NodeIdType id) { return !thisNodes.count(id); });
}

NodeIdType SynchronizingCyclesFinderGraph::getOrAddNode(mlir::Operation *op) {
  if (auto it = opToNodeId.find(op); it != opToNodeId.end())
    return it->second;
  NodeIdType id = addNode(op);
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
    assert(producer && "CFDFC channel must have a defining operation");

    for (mlir::Operation *consumer : channel.getUsers()) {
      assert(opToNodeId.count(producer) &&
             "CFDFC channel producer must be in CFDFC units");
      assert(opToNodeId.count(consumer) &&
             "CFDFC channel consumer must be in CFDFC units");
      NodeIdType srcId = opToNodeId[producer];
      NodeIdType dstId = opToNodeId[consumer];
      addEdge(srcId, dstId, channel);
    }
  }
}

std::string
SynchronizingCyclesFinderGraph::getNodeLabel(NodeIdType nodeId) const {
  return nodes[nodeId].op->getName().getStringRef().str();
}

std::string
SynchronizingCyclesFinderGraph::getNodeDotId(NodeIdType nodeId) const {
  return "node_" + std::to_string(nodeId);
}

void SynchronizingCyclesFinderGraph::computeSccsAndBuildNonCyclicSubgraph() {
  if (nodes.size() == 0)
    return;

  /// Kosaraju's algorithm for SCCs
  /// NOTE: Similar SCC code exists in
  /// experimental/Transforms/ResourceSharing/SharingSupport.cpp
  ///  Kept separate because we need the non-cyclic adjacency list.

  // DFS to compute finishing order
  std::vector<bool> visited(/*count=*/nodes.size(), /*initialValue*/ false);
  std::stack<NodeIdType> finishOrder;

  std::function<void(NodeIdType)> forwardDfs = [&](NodeIdType currentNode) {
    visited[currentNode] = true;
    for (size_t edgeIdx : adjList[currentNode]) {
      NodeIdType successorNode = edges[edgeIdx].dstId;
      if (!visited[successorNode]) {
        forwardDfs(successorNode);
      }
    }
    finishOrder.push(currentNode);
  };

  for (NodeIdType i = 0; i < nodes.size(); ++i) {
    if (!visited[i]) {
      forwardDfs(i);
    }
  }

  // DFS on reverse graph in finish order to find SCCs
  std::vector<size_t> nodeSccId(nodes.size(), 0);
  std::fill(visited.begin(), visited.end(), false);
  size_t sccCount = 0;

  std::function<void(NodeIdType, size_t)> reverseDfs =
      [&](NodeIdType currentNode, size_t sccId) {
        visited[currentNode] = true;
        nodeSccId[currentNode] = sccId;
        for (size_t edgeIdx : revAdjList[currentNode]) {
          NodeIdType successorNode = edges[edgeIdx].srcId;
          if (!visited[successorNode]) {
            reverseDfs(successorNode, sccId);
          }
        }
      };

  while (!finishOrder.empty()) {
    NodeIdType currentNode = finishOrder.top();
    finishOrder.pop();
    if (!visited[currentNode]) {
      reverseDfs(currentNode, sccCount);
      sccCount++;
    }
  }

  // Build non-cyclic adjacency list (only edges between different SCCs)
  nonCyclicAdjList.resize(nodes.size());
  for (size_t edgeIdx = 0; edgeIdx < edges.size(); ++edgeIdx) {
    const auto &edge = edges[edgeIdx];
    if (nodeSccId[edge.srcId] != nodeSccId[edge.dstId]) {
      nonCyclicAdjList[edge.srcId].push_back(edgeIdx);
    }
  }

  LLVM_DEBUG(llvm::errs() << "Computed " << sccCount << " SCCs in CFDFC with "
                          << nodes.size() << " nodes. Non-cyclic subgraph has "
                          << std::accumulate(nonCyclicAdjList.begin(),
                                             nonCyclicAdjList.end(), 0UL,
                                             [](size_t sum, const auto &v) {
                                               return sum + v.size();
                                             })
                          << " edges.\n");
}

std::vector<NodeIdType> SynchronizingCyclesFinderGraph::getAllJoins() const {
  std::vector<NodeIdType> joins;
  for (NodeIdType i = 0; i < nodes.size(); ++i) {
    if (isJoinNode(i)) {
      joins.push_back(i);
    }
  }
  return joins;
}

std::vector<size_t>
SynchronizingCyclesFinderGraph::findEdgesToJoin(const SimpleCycle &cycle,
                                                NodeIdType joinId) const {
  std::set<NodeIdType> cycleNodes(cycle.nodes.begin(), cycle.nodes.end());

  // Step 1: Find all nodes reachable from cycle via non-cyclic edges (forward)
  std::vector<bool> reachableFromCycle(nodes.size(), false);
  std::queue<NodeIdType> fwdQueue;

  // Seed with cycle nodes
  for (NodeIdType nodeId : cycleNodes) {
    reachableFromCycle[nodeId] = true;
    fwdQueue.push(nodeId);
  }

  while (!fwdQueue.empty()) {
    NodeIdType current = fwdQueue.front();
    fwdQueue.pop();
    for (size_t edgeIdx : nonCyclicAdjList[current]) {
      NodeIdType neighbor = edges[edgeIdx].dstId;
      if (!reachableFromCycle[neighbor]) {
        reachableFromCycle[neighbor] = true;
        fwdQueue.push(neighbor);
      }
    }
  }

  // Early exit if join is not reachable from cycle
  if (!reachableFromCycle[joinId])
    return {};

  // Step 2: Find all nodes that can reach join via non-cyclic edges (backward)
  std::vector<bool> canReachJoin(nodes.size(), false);
  std::queue<NodeIdType> bwdQueue;
  canReachJoin[joinId] = true;
  bwdQueue.push(joinId);

  while (!bwdQueue.empty()) {
    NodeIdType current = bwdQueue.front();
    bwdQueue.pop();
    for (size_t edgeIdx : revAdjList[current]) {
      NodeIdType predecessor = edges[edgeIdx].srcId;
      // Only follow non-cyclic edges
      // Check if this edge is in nonCyclicAdjList by checking SCC membership
      if (!canReachJoin[predecessor]) {
        // Check if edge is non-cyclic by seeing if it's in nonCyclicAdjList
        for (size_t ncEdgeIdx : nonCyclicAdjList[predecessor]) {
          if (edges[ncEdgeIdx].dstId == current) {
            canReachJoin[predecessor] = true;
            bwdQueue.push(predecessor);
            break;
          }
        }
      }
    }
  }

  // Step 3: Collect all non-cyclic edges where src is reachable from cycle
  // AND dst can reach join
  std::vector<size_t> edgesOnPath;
  for (NodeIdType nodeId = 0; nodeId < nodes.size(); ++nodeId) {
    if (!reachableFromCycle[nodeId])
      continue;
    for (size_t edgeIdx : nonCyclicAdjList[nodeId]) {
      NodeIdType dst = edges[edgeIdx].dstId;
      if (canReachJoin[dst]) {
        edgesOnPath.push_back(edgeIdx);
      }
    }
  }

  return edgesOnPath;
}

// From:
// [Xu, Josipović, FPGA'24 (https://dl.acm.org/doi/10.1145/3626202.36375)]
//
// Definition 3: Two cycles are a pair of synchronizing cycles in a
// dataflow circuit if the following properties hold: (1) The two cycles
// are disjoint (i.e. they do not have any common units) and belong to
// the same CFC (defined in Section 3.2). (2) There exists at least one
// join that is reachable from both cycles without crossing any edge
// on the cycle in the CFC they belong to.
//
// Algorithm:
// 1. Compute the SCCs of the CFDFC so we can build the non-cyclic subgraph
// required to find paths to joins.
// 2. Find all cycles in the CFDFC.
// 3. For each cycle pair, check if they're disjoint and both can reach a join
// via non-cyclic paths.
// 4. If the criteria are met, get the paths to the joins and add the pair to
// the list of synchronizing cycle pairs.
//
std::vector<SynchronizingCyclePair>
SynchronizingCyclesFinderGraph::findSynchronizingCyclePairs() {
  computeSccsAndBuildNonCyclicSubgraph();

  std::vector<SimpleCycle> allCycles = findAllCycles();
  LLVM_DEBUG(llvm::errs() << "Found " << allCycles.size()
                          << " cycles in CFDFC.\n");

  std::vector<NodeIdType> allJoins = getAllJoins();
  LLVM_DEBUG(llvm::errs() << "Found " << allJoins.size()
                          << " joins in CFDFC.\n");

  std::vector<SynchronizingCyclePair> pairs;

  for (size_t i = 0; i < allCycles.size(); ++i) {
    for (size_t j = i + 1; j < allCycles.size(); ++j) {
      const auto &cycleOne = allCycles[i];
      const auto &cycleTwo = allCycles[j];

      // Criteria 1: Cycles must be disjoint
      if (!cycleOne.isDisjointFrom(cycleTwo)) {
        continue;
      }

      // Criteria 2: Find joins reachable from BOTH cycles via non-cyclic edges
      std::vector<EdgesToJoin> edgesToJoins;

      for (NodeIdType joinId : allJoins) {
        auto edgesOne = findEdgesToJoin(cycleOne, joinId);
        auto edgesTwo = findEdgesToJoin(cycleTwo, joinId);

        // Both cycles must be able to reach this join
        if (!edgesOne.empty() && !edgesTwo.empty()) {
          EdgesToJoin edgeInfo(joinId);
          edgeInfo.edgesFromCycleOne = std::move(edgesOne);
          edgeInfo.edgesFromCycleTwo = std::move(edgesTwo);
          edgesToJoins.push_back(std::move(edgeInfo));
        }
      }

      if (!edgesToJoins.empty()) {
        pairs.emplace_back(cycleOne, cycleTwo, std::move(edgesToJoins));
      }
    }
  }

  LLVM_DEBUG(llvm::errs() << "Found " << pairs.size()
                          << " synchronizing cycle pairs.\n");
  return pairs;
}

std::vector<SimpleCycle> SynchronizingCyclesFinderGraph::findAllCycles() const {
  // Build a Boost graph from our adjacency list
  BoostDataflowSubgraph boostGraph(nodes.size());

  for (const auto &edge : edges) {
    boost::add_edge(edge.srcId, edge.dstId, boostGraph);
  }

  std::vector<SimpleCycle> allCycles;
  CycleCollector collector{allCycles};

  boost::tiernan_all_cycles(boostGraph, collector);

  return allCycles;
}

// [START AI-generated code]

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

  std::set<NodeIdType> cycleOneNodes(pair.cycleOne.nodes.begin(),
                                     pair.cycleOne.nodes.end());
  std::set<NodeIdType> cycleTwoNodes(pair.cycleTwo.nodes.begin(),
                                     pair.cycleTwo.nodes.end());

  std::set<NodeIdType> intermediateNodesOne, intermediateNodesTwo;
  for (const auto &edgeInfo : pair.edgesToJoins) {
    for (size_t edgeIdx : edgeInfo.edgesFromCycleOne) {
      NodeIdType src = edges[edgeIdx].srcId;
      NodeIdType dst = edges[edgeIdx].dstId;
      if (!cycleOneNodes.count(src) && src != edgeInfo.joinId)
        intermediateNodesOne.insert(src);
      if (!cycleOneNodes.count(dst) && dst != edgeInfo.joinId)
        intermediateNodesOne.insert(dst);
    }
    for (size_t edgeIdx : edgeInfo.edgesFromCycleTwo) {
      NodeIdType src = edges[edgeIdx].srcId;
      NodeIdType dst = edges[edgeIdx].dstId;
      if (!cycleTwoNodes.count(src) && src != edgeInfo.joinId)
        intermediateNodesTwo.insert(src);
      if (!cycleTwoNodes.count(dst) && dst != edgeInfo.joinId)
        intermediateNodesTwo.insert(dst);
    }
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

  for (NodeIdType nodeId : cycleOneNodes) {
    file << "    c1_" << getNodeDotId(nodeId) << " [label=\""
         << getNodeLabel(nodeId)
         << "\", style=filled, fillcolor=\"#a5d6a7\", color=\"#2e7d32\"];\n";
  }
  file << "\n";
  for (size_t i = 0; i < pair.cycleOne.nodes.size(); ++i) {
    NodeIdType src = pair.cycleOne.nodes[i];
    NodeIdType dst = pair.cycleOne.nodes[(i + 1) % pair.cycleOne.nodes.size()];
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

  for (NodeIdType nodeId : cycleTwoNodes) {
    file << "    c2_" << getNodeDotId(nodeId) << " [label=\""
         << getNodeLabel(nodeId)
         << "\", style=filled, fillcolor=\"#90caf9\", color=\"#1565c0\"];\n";
  }
  file << "\n";
  for (size_t i = 0; i < pair.cycleTwo.nodes.size(); ++i) {
    NodeIdType src = pair.cycleTwo.nodes[i];
    NodeIdType dst = pair.cycleTwo.nodes[(i + 1) % pair.cycleTwo.nodes.size()];
    file << "    c2_" << getNodeDotId(src) << " -> c2_" << getNodeDotId(dst)
         << " [color=\"#1565c0\", penwidth=2];\n";
  }
  file << "  }\n\n";

  // Intermediate nodes from Cycle One paths (light green)
  if (!intermediateNodesOne.empty()) {
    file << "  // Intermediate nodes from Cycle One to joins\n";
    for (NodeIdType nodeId : intermediateNodesOne) {
      file << "  int1_" << getNodeDotId(nodeId) << " [label=\""
           << getNodeLabel(nodeId)
           << "\", style=filled, fillcolor=\"#c8e6c9\", color=\"#66bb6a\"];\n";
    }
    file << "\n";
  }

  // Intermediate nodes from Cycle Two paths (light blue)
  if (!intermediateNodesTwo.empty()) {
    file << "  // Intermediate nodes from Cycle Two to joins\n";
    for (NodeIdType nodeId : intermediateNodesTwo) {
      file << "  int2_" << getNodeDotId(nodeId) << " [label=\""
           << getNodeLabel(nodeId)
           << "\", style=filled, fillcolor=\"#bbdefb\", color=\"#42a5f5\"];\n";
    }
    file << "\n";
  }

  // Common Joins cluster (orange)
  if (!pair.edgesToJoins.empty()) {
    file << "  subgraph cluster_common_joins {\n";
    file << "    label=\"Common Joins (" << pair.edgesToJoins.size()
         << ")\";\n";
    file << "    style=filled;\n";
    file << "    color=\"#e65100\";\n";
    file << "    fillcolor=\"#fff3e0\";\n";
    file << "    fontcolor=\"#bf360c\";\n";
    file << "    fontsize=12;\n\n";

    for (const auto &edgeInfo : pair.edgesToJoins) {
      file << "    join_" << getNodeDotId(edgeInfo.joinId) << " [label=\""
           << getNodeLabel(edgeInfo.joinId)
           << "\", style=filled, fillcolor=\"#ffcc80\", color=\"#e65100\", "
              "penwidth=2];\n";
    }
    file << "  }\n\n";

    // Draw edges from cycles to joins
    file << "  // Edges from cycles to joins (via non-cyclic subgraph)\n";
    std::set<std::pair<std::string, std::string>> drawnEdges;

    for (const auto &edgeInfo : pair.edgesToJoins) {
      // Edges from Cycle One
      for (size_t edgeIdx : edgeInfo.edgesFromCycleOne) {
        NodeIdType src = edges[edgeIdx].srcId;
        NodeIdType dst = edges[edgeIdx].dstId;
        std::string srcId = cycleOneNodes.count(src)
                                ? "c1_" + getNodeDotId(src)
                                : "int1_" + getNodeDotId(src);
        std::string dstId = (dst == edgeInfo.joinId)
                                ? "join_" + getNodeDotId(dst)
                                : "int1_" + getNodeDotId(dst);
        if (drawnEdges.insert({srcId, dstId}).second) {
          file << "  " << srcId << " -> " << dstId
               << " [color=\"#2e7d32\", style=dashed, penwidth=1.5];\n";
        }
      }

      // Edges from Cycle Two
      for (size_t edgeIdx : edgeInfo.edgesFromCycleTwo) {
        NodeIdType src = edges[edgeIdx].srcId;
        NodeIdType dst = edges[edgeIdx].dstId;
        std::string srcId = cycleTwoNodes.count(src)
                                ? "c2_" + getNodeDotId(src)
                                : "int2_" + getNodeDotId(src);
        std::string dstId = (dst == edgeInfo.joinId)
                                ? "join_" + getNodeDotId(dst)
                                : "int2_" + getNodeDotId(dst);
        if (drawnEdges.insert({srcId, dstId}).second) {
          file << "  " << srcId << " -> " << dstId
               << " [color=\"#1565c0\", style=dashed, penwidth=1.5];\n";
        }
      }
    }
  }

  file << "}\n";
  file.close();
  llvm::errs() << "Dumped synchronizing cycle pair to " << fullPath << "\n";
}

void SynchronizingCyclesFinderGraph::dumpAllSynchronizingCyclePairs(
    llvm::ArrayRef<SynchronizingCyclePair> pairs,
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

    std::set<NodeIdType> cycleOneNodes(pair.cycleOne.nodes.begin(),
                                       pair.cycleOne.nodes.end());
    std::set<NodeIdType> cycleTwoNodes(pair.cycleTwo.nodes.begin(),
                                       pair.cycleTwo.nodes.end());

    // Collect intermediate nodes from edges
    std::set<NodeIdType> intermediateNodesOne, intermediateNodesTwo;
    for (const auto &edgeInfo : pair.edgesToJoins) {
      for (size_t edgeIdx : edgeInfo.edgesFromCycleOne) {
        NodeIdType src = edges[edgeIdx].srcId;
        NodeIdType dst = edges[edgeIdx].dstId;
        if (!cycleOneNodes.count(src) && src != edgeInfo.joinId)
          intermediateNodesOne.insert(src);
        if (!cycleOneNodes.count(dst) && dst != edgeInfo.joinId)
          intermediateNodesOne.insert(dst);
      }
      for (size_t edgeIdx : edgeInfo.edgesFromCycleTwo) {
        NodeIdType src = edges[edgeIdx].srcId;
        NodeIdType dst = edges[edgeIdx].dstId;
        if (!cycleTwoNodes.count(src) && src != edgeInfo.joinId)
          intermediateNodesTwo.insert(src);
        if (!cycleTwoNodes.count(dst) && dst != edgeInfo.joinId)
          intermediateNodesTwo.insert(dst);
      }
    }

    file << "  subgraph cluster_pair_" << pairIdx << " {\n";
    file << "    label=\"Pair " << pairIdx << " (" << pair.edgesToJoins.size()
         << " common joins)\";\n";
    file << "    style=rounded;\n";
    file << "    color=\"#424242\";\n";
    file << "    bgcolor=\"#fafafa\";\n";
    file << "    fontsize=14;\n\n";

    // Cycle One
    file << "    subgraph cluster_" << prefix << "cycle_one {\n";
    file << "      label=\"Cycle One (" << cycleOneNodes.size()
         << " nodes)\";\n";
    file << "      style=filled;\n";
    file << "      color=\"#2e7d32\";\n";
    file << "      fillcolor=\"#e8f5e9\";\n";
    file << "      fontcolor=\"#1b5e20\";\n\n";

    for (NodeIdType nodeId : cycleOneNodes) {
      file << "      " << prefix << "c1_" << getNodeDotId(nodeId)
           << " [label=\"" << getNodeLabel(nodeId)
           << "\", style=filled, fillcolor=\"#a5d6a7\", color=\"#2e7d32\"];\n";
    }
    file << "\n";
    for (size_t i = 0; i < pair.cycleOne.nodes.size(); ++i) {
      NodeIdType src = pair.cycleOne.nodes[i];
      NodeIdType dst =
          pair.cycleOne.nodes[(i + 1) % pair.cycleOne.nodes.size()];
      file << "      " << prefix << "c1_" << getNodeDotId(src) << " -> "
           << prefix << "c1_" << getNodeDotId(dst)
           << " [color=\"#2e7d32\", penwidth=2];\n";
    }
    file << "    }\n\n";

    // Cycle Two
    file << "    subgraph cluster_" << prefix << "cycle_two {\n";
    file << "      label=\"Cycle Two (" << cycleTwoNodes.size()
         << " nodes)\";\n";
    file << "      style=filled;\n";
    file << "      color=\"#1565c0\";\n";
    file << "      fillcolor=\"#e3f2fd\";\n";
    file << "      fontcolor=\"#0d47a1\";\n\n";

    for (NodeIdType nodeId : cycleTwoNodes) {
      file << "      " << prefix << "c2_" << getNodeDotId(nodeId)
           << " [label=\"" << getNodeLabel(nodeId)
           << "\", style=filled, fillcolor=\"#90caf9\", color=\"#1565c0\"];\n";
    }
    file << "\n";
    for (size_t i = 0; i < pair.cycleTwo.nodes.size(); ++i) {
      NodeIdType src = pair.cycleTwo.nodes[i];
      NodeIdType dst =
          pair.cycleTwo.nodes[(i + 1) % pair.cycleTwo.nodes.size()];
      file << "      " << prefix << "c2_" << getNodeDotId(src) << " -> "
           << prefix << "c2_" << getNodeDotId(dst)
           << " [color=\"#1565c0\", penwidth=2];\n";
    }
    file << "    }\n\n";

    // Intermediate nodes from Cycle One
    if (!intermediateNodesOne.empty()) {
      file << "    // Intermediate nodes from Cycle One\n";
      for (NodeIdType nodeId : intermediateNodesOne) {
        file
            << "    " << prefix << "int1_" << getNodeDotId(nodeId)
            << " [label=\"" << getNodeLabel(nodeId)
            << "\", style=filled, fillcolor=\"#c8e6c9\", color=\"#66bb6a\"];\n";
      }
      file << "\n";
    }

    // Intermediate nodes from Cycle Two
    if (!intermediateNodesTwo.empty()) {
      file << "    // Intermediate nodes from Cycle Two\n";
      for (NodeIdType nodeId : intermediateNodesTwo) {
        file
            << "    " << prefix << "int2_" << getNodeDotId(nodeId)
            << " [label=\"" << getNodeLabel(nodeId)
            << "\", style=filled, fillcolor=\"#bbdefb\", color=\"#42a5f5\"];\n";
      }
      file << "\n";
    }

    // Common Joins
    if (!pair.edgesToJoins.empty()) {
      file << "    subgraph cluster_" << prefix << "joins {\n";
      file << "      label=\"Common Joins\";\n";
      file << "      style=filled;\n";
      file << "      color=\"#e65100\";\n";
      file << "      fillcolor=\"#fff3e0\";\n";
      file << "      fontcolor=\"#bf360c\";\n\n";

      for (const auto &edgeInfo : pair.edgesToJoins) {
        file << "      " << prefix << "join_" << getNodeDotId(edgeInfo.joinId)
             << " [label=\"" << getNodeLabel(edgeInfo.joinId)
             << "\", style=filled, fillcolor=\"#ffcc80\", color=\"#e65100\", "
                "penwidth=2];\n";
      }
      file << "    }\n\n";

      // Draw edges (deduplicated)
      std::set<std::pair<std::string, std::string>> drawnEdges;

      for (const auto &edgeInfo : pair.edgesToJoins) {
        // Edges from Cycle One
        for (size_t edgeIdx : edgeInfo.edgesFromCycleOne) {
          NodeIdType src = edges[edgeIdx].srcId;
          NodeIdType dst = edges[edgeIdx].dstId;
          std::string srcId = cycleOneNodes.count(src)
                                  ? prefix + "c1_" + getNodeDotId(src)
                                  : prefix + "int1_" + getNodeDotId(src);
          std::string dstId = (dst == edgeInfo.joinId)
                                  ? prefix + "join_" + getNodeDotId(dst)
                                  : prefix + "int1_" + getNodeDotId(dst);
          if (drawnEdges.insert({srcId, dstId}).second) {
            file << "    " << srcId << " -> " << dstId
                 << " [color=\"#2e7d32\", style=dashed, penwidth=1.5];\n";
          }
        }

        // Edges from Cycle Two
        for (size_t edgeIdx : edgeInfo.edgesFromCycleTwo) {
          NodeIdType src = edges[edgeIdx].srcId;
          NodeIdType dst = edges[edgeIdx].dstId;
          std::string srcId = cycleTwoNodes.count(src)
                                  ? prefix + "c2_" + getNodeDotId(src)
                                  : prefix + "int2_" + getNodeDotId(src);
          std::string dstId = (dst == edgeInfo.joinId)
                                  ? prefix + "join_" + getNodeDotId(dst)
                                  : prefix + "int2_" + getNodeDotId(dst);
          if (drawnEdges.insert({srcId, dstId}).second) {
            file << "    " << srcId << " -> " << dstId
                 << " [color=\"#1565c0\", style=dashed, penwidth=1.5];\n";
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

// [END AI-generated code]

} // namespace dynamatic
