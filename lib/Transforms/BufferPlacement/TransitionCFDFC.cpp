//===- TransitionCFDFC.cpp - Transition-based CFDFC ---------------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/TransitionCFDFC.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;

std::string TransitionNode::getDotId() const {
  return "node_" + std::to_string(id);
}

std::string TransitionNode::getLabel() const {
  std::string opName = op->getName().getStringRef().str();
  // Simplified label: OpName\nStep: X
  return opName + "\\nStep: " + std::to_string(step);
}

TransitionCFDFC::TransitionCFDFC(handshake::FuncOp funcOp,
                                 const std::vector<ArchBB> &sequence)
    : funcOp(funcOp) {

  if (sequence.empty())
    return;

  std::vector<unsigned> archSrcStep(sequence.size());
  std::vector<unsigned> archDstStep(sequence.size());

  unsigned currentStep = 0;
  stepToBB[currentStep] = sequence[0].srcBB;
  stepToBB[currentStep + 1] = sequence[0].dstBB;
  archSrcStep[0] = currentStep;
  archDstStep[0] = currentStep + 1;
  unsigned maxStep = currentStep + 1;

  for (size_t i = 1; i < sequence.size(); ++i) {
    const ArchBB &prev = sequence[i - 1];
    const ArchBB &curr = sequence[i];

    if (prev.dstBB == curr.srcBB) {
      archSrcStep[i] = archDstStep[i - 1];
      maxStep++;
      archDstStep[i] = maxStep;
      stepToBB[maxStep] = curr.dstBB;
    } else {
      maxStep++;
      archSrcStep[i] = maxStep;
      stepToBB[maxStep] = curr.srcBB;

      maxStep++;
      archDstStep[i] = maxStep;
      stepToBB[maxStep] = curr.dstBB;
    }
  }

  LogicBBs logicBBs = getLogicBBs(funcOp);
  for (auto const &[step, bb] : stepToBB) {
    if (logicBBs.blocks.count(bb)) {
      for (Operation *op : logicBBs.blocks[bb]) {
        getOrAddNode(op, step);
      }
    }
  }

  for (const auto &node : nodes) {
    unsigned step = node.step;
    Operation *u = node.op;
    for (Value res : u->getResults()) {
      for (Operation *v : res.getUsers()) {
        if (nodeMap.count({v, step})) {
          addEdge(node.id, nodeMap[{v, step}], res);
        }
      }
    }
  }

  for (size_t i = 0; i < sequence.size(); ++i) {
    unsigned srcStep = archSrcStep[i];
    unsigned dstStep = archDstStep[i];

    for (size_t uId = 0; uId < nodes.size(); ++uId) {
      if (nodes[uId].step != srcStep)
        continue;

      Operation *u = nodes[uId].op;
      for (Value res : u->getResults()) {
        for (Operation *v : res.getUsers()) {
          if (nodeMap.count({v, dstStep})) {
            addEdge(uId, nodeMap[{v, dstStep}], res);
          }
        }
      }
    }
  }
}

size_t TransitionCFDFC::getOrAddNode(Operation *op, unsigned step) {
  if (nodeMap.count({op, step}))
    return nodeMap[{op, step}];

  size_t id = nodes.size();
  nodes.emplace_back(op, step, id);
  adjList.emplace_back();
  nodeMap[{op, step}] = id;
  return id;
}

void TransitionCFDFC::addEdge(size_t srcId, size_t dstId, Value channel) {
  adjList[srcId].emplace_back(dstId, channel);
}

void TransitionCFDFC::runDFS() {
  llvm::errs() << "Running DFS on TransitionCFDFC (Nodes: " << nodes.size()
               << ")...\n";
  llvm::errs().flush(); // FORCE FLUSH
  std::vector<bool> visited(nodes.size(), false);

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!visited[i]) {
      dfsVisit(i, visited, llvm::errs());
    }
  }
  llvm::errs().flush(); // FORCE FLUSH
}

void TransitionCFDFC::dfsVisit(size_t u, std::vector<bool> &visited,
                               llvm::raw_ostream &os) {
  visited[u] = true;
  TransitionNode &node = nodes[u];
  os << "Visited: " << node.getLabel() << "\n";

  for (auto &edge : adjList[u]) {
    size_t v = edge.first;
    if (!visited[v]) {
      dfsVisit(v, visited, os);
    }
  }
}

void TransitionCFDFC::dumpGraphViz(StringRef filename) {
  // Resolve absolute path
  SmallString<256> fullPath;
  if (auto ec = llvm::sys::fs::current_path(fullPath)) {
    llvm::errs() << "Failed to get current path: " << ec.message() << "\n";
    // Fallback to filename as is
    fullPath = filename;
  } else {
    llvm::sys::path::append(fullPath, filename);
  }

  std::string pathStr = fullPath.str().str();
  std::ofstream file(pathStr);

  if (!file.is_open()) {
    llvm::errs() << "Could not open file: " << pathStr << "\n";
    return;
  }

  file << "digraph TransitionCFDFC {\n";
  file << "  rankdir=TB;\n";
  file << "  compound=true;\n"; // Enable cluster-to-cluster edges if needed

  // Group nodes by step to create clusters
  std::map<unsigned, std::vector<const TransitionNode *>> nodesByStep;
  for (const auto &node : nodes) {
    nodesByStep[node.step].push_back(&node);
  }

  // Create a subgraph for each step
  for (const auto &[step, stepNodes] : nodesByStep) {
    unsigned bbID = stepToBB.count(step) ? stepToBB.at(step) : 999;

    file << "  subgraph cluster_" << step << " {\n";
    file << "    label=\"Step " << step << " (BB " << bbID << ")\";\n";
    file << "    style=dashed;\n";
    file << "    color=black;\n";

    for (const auto *node : stepNodes) {
      file << "    " << node->getDotId() << " [label=\"" << node->getLabel()
           << "\"];\n";
    }
    file << "  }\n";
  }

  // Edges
  for (size_t u = 0; u < adjList.size(); ++u) {
    for (const auto &edge : adjList[u]) {
      size_t v = edge.first;
      file << "  " << nodes[u].getDotId() << " -> " << nodes[v].getDotId()
           << ";\n";
    }
  }

  file << "}\n";
  file.close();

  llvm::errs() << "Dumped graph to " << pathStr << "\n";
  llvm::errs().flush();
}

void TransitionCFDFC::buildReverseAdjList() {
  if (!revAdjList.empty())
    return;
  revAdjList.resize(nodes.size());
  for (size_t u = 0; u < adjList.size(); ++u) {
    for (const auto &edge : adjList[u]) {
      revAdjList[edge.first].push_back(u);
    }
  }
}

void TransitionCFDFC::findAndDumpReconvergentPaths(StringRef outputDir) {
  buildReverseAdjList();

  std::vector<size_t> forks;
  std::vector<size_t> joins;

  // Identify Forks and Joins
  for (size_t i = 0; i < nodes.size(); ++i) {
    Operation *op = nodes[i].op;
    if (isa<handshake::ForkOp>(op) || isa<handshake::LazyForkOp>(op)) {
      forks.push_back(i);
    } else if (isa<handshake::MergeOp>(op) ||
               isa<handshake::ControlMergeOp>(op) ||
               isa<handshake::MuxOp>(op)) {
      // Treating Mux as a join point as well, usually Merges are the main ones.
      joins.push_back(i);
    }
  }

  llvm::errs() << "Found " << forks.size() << " forks and " << joins.size()
               << " joins.\n";

  // Prepare single output file
  std::string filename = (outputDir + "/reconvergent_paths.dot").str();
  SmallString<256> fullPath;
  if (llvm::sys::path::is_absolute(filename)) {
    fullPath = filename;
  } else {
    llvm::sys::fs::current_path(fullPath);
    llvm::sys::path::append(fullPath, filename);
  }

  std::ofstream file(fullPath.c_str());
  if (!file.is_open()) {
    llvm::errs() << "Failed to open " << fullPath << "\n";
    return;
  }

  file << "digraph ReconvergentPaths {\n";
  file << "  rankdir=TB;\n";
  file << "  compound=true;\n";

  // Set to track unique sets of nodes to avoid duplicates
  std::set<std::vector<size_t>> seenIntersections;
  int pathCount = 0;

  for (size_t fork : forks) {
    // Forward reachability from fork
    std::vector<bool> reachableFromFork(nodes.size(), false);
    std::queue<size_t> q;
    q.push(fork);
    reachableFromFork[fork] = true;
    while (!q.empty()) {
      size_t u = q.front();
      q.pop();
      for (auto &edge : adjList[u]) {
        if (!reachableFromFork[edge.first]) {
          reachableFromFork[edge.first] = true;
          q.push(edge.first);
        }
      }
    }

    for (size_t join : joins) {
      if (!reachableFromFork[join])
        continue;

      // Backward reachability from join
      std::vector<bool> canReachJoin(nodes.size(), false);
      std::queue<size_t> bq;
      bq.push(join);
      canReachJoin[join] = true;
      while (!bq.empty()) {
        size_t u = bq.front();
        bq.pop();
        for (size_t v : revAdjList[u]) {
          if (!canReachJoin[v]) {
            canReachJoin[v] = true;
            bq.push(v);
          }
        }
      }

      // Intersection
      std::vector<size_t> intersection;
      for (size_t i = 0; i < nodes.size(); ++i) {
        if (reachableFromFork[i] && canReachJoin[i]) {
          intersection.push_back(i);
        }
      }

      // Sort intersection for consistent set comparison
      std::sort(intersection.begin(), intersection.end());

      // Filter trivial paths (size <= 2) and duplicates
      if (intersection.size() > 2) {
        if (seenIntersections.count(intersection)) {
          continue; // Skip duplicate
        }
        seenIntersections.insert(intersection);

        // Create a subgraph for this reconvergent structure
        file << "  subgraph cluster_" << pathCount << " {\n";
        file << "    label=\"Reconvergent Path " << pathCount << " (Fork "
             << fork << " -> Join " << join << ")\";\n";
        file << "    style=rounded;\n";
        file << "    color=blue;\n";

        std::set<size_t> validNodes(intersection.begin(), intersection.end());

        // Add nodes
        for (size_t idx : intersection) {
          std::string uniqueNodeID =
              "path" + std::to_string(pathCount) + "_" + nodes[idx].getDotId();
          file << "    " << uniqueNodeID << " [label=\""
               << nodes[idx].getLabel() << "\"];\n";
        }

        // Add edges
        for (size_t u : intersection) {
          std::string uID =
              "path" + std::to_string(pathCount) + "_" + nodes[u].getDotId();
          for (const auto &edge : adjList[u]) {
            if (validNodes.count(edge.first)) {
              std::string vID = "path" + std::to_string(pathCount) + "_" +
                                nodes[edge.first].getDotId();
              file << "    " << uID << " -> " << vID << ";\n";
            }
          }
        }

        file << "  }\n"; // End subgraph
        pathCount++;
      }
    }
  }

  file << "}\n";
  file.close();
  llvm::errs() << "Dumped " << pathCount << " unique reconvergent paths to "
               << fullPath << "\n";
}