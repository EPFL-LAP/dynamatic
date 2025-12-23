//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class that enumerates reconvergent paths in a dataflow graph
// from a transition sequence of length n. Implementation is basef on:
// [Xu, Josipović, FPGA'24 (https://dl.acm.org/doi/10.1145/3626202.36375)]
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/DataflowGraph/ReconvergentPathFinder.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DataflowGraph/DataflowGraphBase.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <fstream>
#include <queue>

namespace dynamatic {

bool ReconvergentPathFinderGraph::isForkNode(size_t nodeId) const {
  return isa<handshake::ForkOp>(nodes[nodeId].op) || isa<handshake::LazyForkOp>(nodes[nodeId].op);
}

bool ReconvergentPathFinderGraph::isJoinNode(size_t nodeId) const {
  return isa<handshake::MuxOp>(nodes[nodeId].op) || isa<handshake::ConditionalBranchOp>(nodes[nodeId].op);
}

std::string ReconvergentPathFinderGraph::getNodeLabel(size_t nodeId) const {
  std::string opName = nodes[nodeId].op->getName().getStringRef().str();
  return opName + "\\nStep: " + std::to_string(getNodeStep(nodeId));
}

std::string ReconvergentPathFinderGraph::getNodeDotId(size_t nodeId) const {
  return "node_" + std::to_string(nodeId);
}

void ReconvergentPathFinderGraph::buildGraphFromSequence(
    handshake::FuncOp funcOp, const std::vector<ArchBB> &sequence) {
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
          addEdge(node.id, nodeMap[{user, step}], result, DataflowGraphEdgeType::INTRA_BB);
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
            addEdge(node.id, nodeMap[{user, dstStep}], result, DataflowGraphEdgeType::INTER_BB);
          }
        }
      }
    }
  }
}

std::vector<ReconvergentPath> ReconvergentPathFinderGraph::findReconvergentPaths() const {
  std::vector<size_t> forks;
  std::vector<size_t> joins;

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (isForkNode(i)) {
      forks.push_back(i);
    } else if (isJoinNode(i)) {
      joins.push_back(i);
    }
  }

  // Track unique node sets we've seen
  std::set<std::set<size_t>> seenNodeSets;
  std::vector<ReconvergentPath> paths;

  for (size_t forkId : forks) {
    // BFS forward from fork to find all reachable nodes
    std::vector<bool> reachableFromFork(nodes.size(), false);
    std::queue<size_t> fwdQueue;
    fwdQueue.push(forkId);
    reachableFromFork[forkId] = true;

    while (!fwdQueue.empty()) {
      size_t u = fwdQueue.front();
      fwdQueue.pop();
      for (size_t edgeId : adjList[u]) {
        size_t v = edges[edgeId].dstId;
        if (!reachableFromFork[v]) {
          reachableFromFork[v] = true;
          fwdQueue.push(v);
        }
      }
    }

    for (size_t joinId : joins) {
      // Skip if join is not reachable from fork
      if (!reachableFromFork[joinId])
        continue;

      // BFS backward from join to find all nodes that can reach it
      std::vector<bool> canReachJoin(nodes.size(), false);
      std::queue<size_t> bwdQueue;
      bwdQueue.push(joinId);
      canReachJoin[joinId] = true;

      while (!bwdQueue.empty()) {
        size_t u = bwdQueue.front();
        bwdQueue.pop();
        for (size_t edgeId : revAdjList[u]) {
          size_t v = edges[edgeId].srcId; 
          if (!canReachJoin[v]) {
            canReachJoin[v] = true;
            bwdQueue.push(v);
          }
        }
      }

      // The fork must have >=2 direct successors that can reach the join.
      // Otherwise it's just a linear chain, not actual divergence/reconvergence.
      unsigned numDivergingPaths = 0;
      for (size_t edgeId : adjList[forkId]) {
        size_t successor = edges[edgeId].dstId;
        if (canReachJoin[successor])
          numDivergingPaths++;
      }

      // Skip if only 0 or 1 path from fork reaches join (not reconvergent)
      if (numDivergingPaths < 2)
        continue;

      // Nodes reachable from fork AND can reach join
      std::set<size_t> intersection;
      for (size_t i = 0; i < nodes.size(); ++i) {
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

  llvm::errs() << "Found " << paths.size() << " reconvergent paths from "
               << forks.size() << " forks and " << joins.size() << " joins.\n";

  return paths;
}

void ReconvergentPathFinderGraph::dumpReconvergentPaths(
    const std::vector<ReconvergentPath> &paths, llvm::StringRef filename) const {
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
    file << "    label=\"Path " << pathIdx << " (Fork: "
         << getNodeLabel(path.forkNodeId) << " -> Join: "
         << getNodeLabel(path.joinNodeId) << ")\";\n";
    file << "    style=rounded;\n";
    file << "    color=blue;\n";
    file << "    bgcolor=\"#e8f4fc\";\n\n";

    // Emit nodes with unique IDs per path to avoid conflicts
    for (size_t nodeId : path.nodeIds) {
      std::string uniqueId = "p" + std::to_string(pathIdx) + "_" +
                             getNodeDotId(nodeId);
      std::string color = "";
      if (nodeId == path.forkNodeId)
        color = ", style=filled, fillcolor=\"#90EE90\""; // Green for fork
      else if (nodeId == path.joinNodeId)
        color = ", style=filled, fillcolor=\"#FFB6C1\""; // Pink for join

      file << "    " << uniqueId << " [label=\"" << getNodeLabel(nodeId)
           << "\"" << color << "];\n";
    }

    file << "\n";

    // Emit edges within this path (different styles for intra/inter-BB)
    for (size_t srcId : path.nodeIds) {
      for (size_t edgeId : adjList[srcId]) {
        auto &edge = edges[edgeId];
        size_t dstId = edge.dstId;
        if (path.nodeIds.count(dstId)) {
          std::string srcUniqueId = "p" + std::to_string(pathIdx) + "_" +
                                    getNodeDotId(srcId);
          std::string dstUniqueId = "p" + std::to_string(pathIdx) + "_" +
                                    getNodeDotId(dstId);
          std::string style = (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "solid" : "dashed";
          std::string color = (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "black" : "blue";
          file << "    " << srcUniqueId << " -> " << dstUniqueId
               << " [style=" << style << ", color=" << color << "];\n";
        }
      }
    }

    file << "  }\n\n";
  }

  file << "}\n";
  file.close();
  llvm::errs() << "Dumped " << paths.size() << " reconvergent paths to "
               << fullPath << "\n";
}

void ReconvergentPathFinderGraph::dumpTransitionGraph(llvm::StringRef filename) const {
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
  std::map<unsigned, std::vector<const DataflowGraphNode<Operation *> *>> nodesByStep;
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
      file << "    " << getNodeDotId(node->id) << " [label=\"" << getNodeLabel(node->id)
           << "\"];\n";
    }
    file << "  }\n\n";
  }

  // Emit edges with different styles
  for (const auto &edge : edges) {
    std::string style = (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "solid" : "dashed";
    std::string color = (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "black" : "blue";

    file << "  " << getNodeDotId(edge.srcId) << " -> "
         << getNodeDotId(edge.dstId) << " [style=" << style
         << ", color=" << color << "];\n";
  }

  file << "}\n";
  file.close();
  llvm::errs() << "Dumped DataflowGraph to " << fullPath << "\n";
}

void ReconvergentPathFinderGraph::dumpAllGraphs(
    const std::vector<ReconvergentPathFinderGraph> &graphs,
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
    std::map<unsigned, std::vector<size_t>> nodesByStep;
    for (const auto &node : graph.getNodes()) {
      nodesByStep[graph.getNodeStep(node.id)].push_back(node.id);
    }

    // Emit nodes grouped by step in subgraph clusters
    for (const auto &[step, stepNodeIds] : nodesByStep) {
      unsigned bbID = graph.getStepBB(step);
      file << "    subgraph cluster_" << graphPrefix << "step_" << step << " {\n";
      file << "      label=\"Step " << step << " (BB " << bbID << ")\";\n";
      file << "      style=solid;\n";
      file << "      color=black;\n";
      file << "      bgcolor=\"#f0f0f0\";\n";

      for (size_t nodeId : stepNodeIds) {
        file << "      " << graphPrefix << graph.getNodeDotId(nodeId)
             << " [label=\"" << graph.getNodeLabel(nodeId) << "\"];\n";
      }
      file << "    }\n\n";
    }

    // Emit edges with different styles
    for (const auto &edge : graph.getEdges()) {
      std::string style =
          (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "solid" : "dashed";
      std::string color =
          (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "black" : "blue";

      file << "    " << graphPrefix << graph.getNodeDotId(edge.srcId) << " -> "
           << graphPrefix << graph.getNodeDotId(edge.dstId) << " [style="
           << style << ", color=" << color << "];\n";
    }

    file << "  }\n\n";
  }

  file << "}\n";
  file.close();
  llvm::errs() << "Dumped " << graphs.size() << " dataflow graphs to "
               << fullPath << "\n";
}

void ReconvergentPathFinderGraph::dumpAllReconvergentPaths(
    const std::vector<
        std::pair<size_t, std::pair<const ReconvergentPathFinderGraph *,
                                    std::vector<ReconvergentPath>>>> &graphPaths,
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

  file << "digraph AllReconvergentPaths {\n";
  file << "  rankdir=TB;\n";
  file << "  bgcolor=white;\n";
  file << "  compound=true;\n\n";

  size_t totalPaths = 0;
  for (const auto &[graphIdx, graphAndPaths] : graphPaths) {
    const ReconvergentPathFinderGraph *graph = graphAndPaths.first;
    const std::vector<ReconvergentPath> &paths = graphAndPaths.second;

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
      for (size_t nodeId : path.nodeIds) {
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
      for (size_t srcId : path.nodeIds) {
        for (size_t edgeId : graph->getAdjList()[srcId]) {
          const auto &edge = graph->getEdges()[edgeId];
          size_t dstId = edge.dstId;
          if (path.nodeIds.count(dstId)) {
            std::string srcUniqueId = uniquePrefix + graph->getNodeDotId(srcId);
            std::string dstUniqueId = uniquePrefix + graph->getNodeDotId(dstId);
            std::string style =
                (edge.type == DataflowGraphEdgeType::INTRA_BB) ? "solid"
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
      totalPaths++;
    }
  }

  file << "}\n";
  file.close();
  llvm::errs() << "Dumped " << totalPaths << " reconvergent paths from "
               << graphPaths.size() << " graphs to " << fullPath << "\n";
}

} // namespace dynamatic