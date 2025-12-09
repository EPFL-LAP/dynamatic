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
