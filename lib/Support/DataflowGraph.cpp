//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A generic dataflow graph representation for circuit analysis & optimization.
//
//===----------------------------------------------------------------------===//

// TODO: Make sure to check for branch operations to add a step artificially

#include "dynamatic/Support/DataflowGraph.h"
#include "dynamatic/Support/CFG.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <fstream>
#include <iostream>
#include <queue>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

std::string DataflowGraphNode::getDotId() const {
  return "node_" + std::to_string(id);
}

std::string DataflowGraphNode::getLabel() const {
  std::string opName = op->getName().getStringRef().str();
  return opName + "\\nStep: " + std::to_string(step);
}

DataflowGraph::DataflowGraph(
    handshake::FuncOp funcOp,
    const std::vector<dynamatic::experimental::ArchBB> &sequence)
    : funcOp(funcOp) {

  if (sequence.empty())
    return;

  LogicBBs logicBBs = getLogicBBs(funcOp);

  // Build step-to-BB mapping and track arch connections.
  // ====================================================
  // We need to track which step each arch's source and destination map to,
  // so that inter-BB edges are only created along the actual arch sequence.

  std::vector<unsigned> archSrcStep(sequence.size());
  std::vector<unsigned> archDstStep(sequence.size());

  unsigned maxStep = 0;
  stepToBB[0] = sequence[0].srcBB;
  stepToBB[1] = sequence[0].dstBB;
  archSrcStep[0] = 0;
  archDstStep[0] = 1;
  maxStep = 1;

  for (size_t i = 1; i < sequence.size(); ++i) {
    const ArchBB &prev = sequence[i - 1];
    const ArchBB &curr = sequence[i];

    if (prev.dstBB == curr.srcBB) {
      // Current arch continues from previous arch's destination
      archSrcStep[i] = archDstStep[i - 1];
      maxStep++;
      archDstStep[i] = maxStep;
      stepToBB[maxStep] = curr.dstBB;
    } else {
      // Current arch starts from a different BB - need new steps for both
      maxStep++;
      archSrcStep[i] = maxStep;
      stepToBB[maxStep] = curr.srcBB;

      maxStep++;
      archDstStep[i] = maxStep;
      stepToBB[maxStep] = curr.dstBB;
    }
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
  // IMPORTANT: Skip backedges - they cross to the next iteration, not same step.

  for (const auto &node : nodes) {
    unsigned step = node.step;
    Operation *op = node.op;

    for (Value result : op->getResults()) {
      // Backedges go to the next iteration (inter-BB), not within same step
      if (isBackedge(result))
        continue;

      for (Operation *user : result.getUsers()) {
        // Check if user exists at the same step
        if (nodeMap.count({user, step})) {
          addEdge(node.id, nodeMap[{user, step}], INTRA_BB, result);
        }
      }
    }
  }

  // Populate inter-BB edges (between steps along arch sequence).
  // ============================================================
  // Only create edges from archSrcStep[i] to archDstStep[i].
  // For self-loops (srcBB == dstBB), only backedge channels cross iterations.

  for (size_t i = 0; i < sequence.size(); ++i) {
    unsigned srcStep = archSrcStep[i];
    unsigned dstStep = archDstStep[i];
    bool isSelfLoop = (sequence[i].srcBB == sequence[i].dstBB);

    for (const auto &node : nodes) {
      if (node.step != srcStep)
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
            addEdge(node.id, nodeMap[{user, dstStep}], INTER_BB, result);
          }
        }
      }
    }
  }
}

size_t DataflowGraph::getOrAddNode(Operation *op, unsigned step) {
  auto key = std::make_pair(op, step);
  if (nodeMap.count(key))
    return nodeMap[key];

  size_t id = nodes.size();
  nodes.emplace_back(op, step, id);
  adjList.emplace_back();
  nodeMap[key] = id;
  return id;
}

void DataflowGraph::addEdge(size_t srcId, size_t dstId,
                            DataflowGraphEdgeType type, Value channel) {
  size_t edgeId = edges.size();
  edges.emplace_back(srcId, dstId, type, channel);
  adjList[srcId].push_back(edgeId);
}

void DataflowGraph::runDFS() {
  llvm::errs() << "Running DFS on DataflowGraph (Nodes: " << nodes.size()
               << ", Edges: " << edges.size() << ")...\n";

  std::vector<bool> visited(nodes.size(), false);
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!visited[i]) {
      dfsVisit(i, visited, llvm::errs());
    }
  }
}

void DataflowGraph::dfsVisit(size_t nodeId, std::vector<bool> &visited,
                             llvm::raw_ostream &os) {
  visited[nodeId] = true;
  const DataflowGraphNode &node = nodes[nodeId];
  os << "Visited: " << node.getLabel() << "\n";

  for (size_t edgeId : adjList[nodeId]) {
    const DataflowGraphEdge &edge = edges[edgeId];
    size_t dstNodeId = edge.dstId;

    if (!visited[dstNodeId]) {
      std::string edgeType =
          (edge.type == INTRA_BB) ? "intra-BB" : "inter-BB";
      os << "  -> " << nodes[dstNodeId].getLabel() << " (" << edgeType << ")\n";
      dfsVisit(dstNodeId, visited, os);
    }
  }
}

void DataflowGraph::dumpGraphViz(llvm::StringRef filename) {
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
    nodesByStep[node.step].push_back(&node);
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
      file << "    " << node->getDotId() << " [label=\"" << node->getLabel()
           << "\"];\n";
    }
    file << "  }\n\n";
  }

  // Emit edges with different styles
  for (const auto &edge : edges) {
    std::string style = (edge.type == INTRA_BB) ? "solid" : "dashed";
    std::string color = (edge.type == INTRA_BB) ? "black" : "blue";

    file << "  " << nodes[edge.srcId].getDotId() << " -> "
         << nodes[edge.dstId].getDotId() << " [style=" << style
         << ", color=" << color << "];\n";
  }

  file << "}\n";
  file.close();
  llvm::errs() << "Dumped DataflowGraph to " << fullPath << "\n";
}

// === Reconvergent Path Analysis === //

void DataflowGraph::buildReverseAdjList() {
  if (!revAdjList.empty())
    return;

  revAdjList.resize(nodes.size());
  for (size_t srcId = 0; srcId < adjList.size(); ++srcId) {
    for (size_t edgeId : adjList[srcId]) {
      size_t dstId = edges[edgeId].dstId;
      revAdjList[dstId].push_back(srcId);
    }
  }
}

std::vector<ReconvergentPath> DataflowGraph::findReconvergentPaths() {
  buildReverseAdjList();

  std::vector<size_t> forks;
  std::vector<size_t> joins;

  for (size_t i = 0; i < nodes.size(); ++i) {
    Operation *op = nodes[i].op;
    if (isa<handshake::ForkOp>(op) || isa<handshake::LazyForkOp>(op)) {
      forks.push_back(i);
    } else if (isa<handshake::MergeOp>(op) ||
               isa<handshake::ControlMergeOp>(op) ||
               isa<handshake::MuxOp>(op)) {
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
        for (size_t v : revAdjList[u]) {
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

void DataflowGraph::dumpReconvergentPaths(
    const std::vector<ReconvergentPath> &paths, llvm::StringRef filename) {
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

  file << "digraph ReconvergentPaths {\n";
  file << "  rankdir=TB;\n";
  file << "  bgcolor=white;\n";
  file << "  compound=true;\n\n";

  for (size_t pathIdx = 0; pathIdx < paths.size(); ++pathIdx) {
    const ReconvergentPath &path = paths[pathIdx];

    file << "  subgraph cluster_path_" << pathIdx << " {\n";
    file << "    label=\"Path " << pathIdx << " (Fork: "
         << nodes[path.forkNodeId].getLabel() << " -> Join: "
         << nodes[path.joinNodeId].getLabel() << ")\";\n";
    file << "    style=rounded;\n";
    file << "    color=blue;\n";
    file << "    bgcolor=\"#e8f4fc\";\n\n";

    // Emit nodes with unique IDs per path to avoid conflicts
    for (size_t nodeId : path.nodeIds) {
      std::string uniqueId = "p" + std::to_string(pathIdx) + "_" +
                             nodes[nodeId].getDotId();
      std::string color = "";
      if (nodeId == path.forkNodeId)
        color = ", style=filled, fillcolor=\"#90EE90\""; // Green for fork
      else if (nodeId == path.joinNodeId)
        color = ", style=filled, fillcolor=\"#FFB6C1\""; // Pink for join

      file << "    " << uniqueId << " [label=\"" << nodes[nodeId].getLabel()
           << "\"" << color << "];\n";
    }

    file << "\n";

    // Emit edges within this path (different styles for intra/inter-BB)
    for (size_t srcId : path.nodeIds) {
      for (size_t edgeId : adjList[srcId]) {
        const DataflowGraphEdge &edge = edges[edgeId];
        size_t dstId = edge.dstId;
        if (path.nodeIds.count(dstId)) {
          std::string srcUniqueId = "p" + std::to_string(pathIdx) + "_" +
                                    nodes[srcId].getDotId();
          std::string dstUniqueId = "p" + std::to_string(pathIdx) + "_" +
                                    nodes[dstId].getDotId();
          std::string style = (edge.type == INTRA_BB) ? "solid" : "dashed";
          std::string color = (edge.type == INTRA_BB) ? "black" : "blue";
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
