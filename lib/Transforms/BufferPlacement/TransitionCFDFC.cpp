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
#include "llvm/Support/Path.h" // Add this include for path manipulation
#include <fstream>
#include <iostream>

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

  // 1. Determine the sequence of steps and associated BBs
  // We track which 'step' (instance index) corresponds to the source and
  // destination of each arch in the sequence.
  std::vector<unsigned> archSrcStep(sequence.size());
  std::vector<unsigned> archDstStep(sequence.size());
  std::map<unsigned, unsigned> stepToBB;

  unsigned currentStep = 0;
  // Initialize with the first arch
  stepToBB[currentStep] = sequence[0].srcBB;
  stepToBB[currentStep + 1] = sequence[0].dstBB;
  archSrcStep[0] = currentStep;
  archDstStep[0] = currentStep + 1;
  unsigned maxStep = currentStep + 1;

  // Process the rest of the sequence
  for (size_t i = 1; i < sequence.size(); ++i) {
    const ArchBB &prev = sequence[i - 1];
    const ArchBB &curr = sequence[i];

    if (prev.dstBB == curr.srcBB) {
      // Connected: reuse the destination step of the previous arch as the
      // source of current
      archSrcStep[i] = archDstStep[i - 1];
      // Create new step for destination
      maxStep++;
      archDstStep[i] = maxStep;
      stepToBB[maxStep] = curr.dstBB;
    } else {
      // Disconnected: start a new chain segment
      maxStep++;
      archSrcStep[i] = maxStep;
      stepToBB[maxStep] = curr.srcBB;

      maxStep++;
      archDstStep[i] = maxStep;
      stepToBB[maxStep] = curr.dstBB;
    }
  }

  // 2. Instantiate nodes for each step
  LogicBBs logicBBs = getLogicBBs(funcOp);
  for (auto const &[step, bb] : stepToBB) {
    if (logicBBs.blocks.count(bb)) {
      for (Operation *op : logicBBs.blocks[bb]) {
        getOrAddNode(op, step);
      }
    }
  }

  // 3. Create edges

  // A. Intra-block edges for each step (same step -> same step)
  for (const auto &node : nodes) {
    unsigned step = node.step;
    Operation *u = node.op;
    for (Value res : u->getResults()) {
      for (Operation *v : res.getUsers()) {
        // If v is in the same step (same BB logic)
        if (nodeMap.count({v, step})) {
          addEdge(node.id, nodeMap[{v, step}], res);
        }
      }
    }
  }

  for (size_t i = 0; i < sequence.size(); ++i) {
    unsigned srcStep = archSrcStep[i];
    unsigned dstStep = archDstStep[i];

    // Iterate over all nodes in the source step
    for (size_t uId = 0; uId < nodes.size(); ++uId) {
      if (nodes[uId].step != srcStep)
        continue;

      Operation *u = nodes[uId].op;
      for (Value res : u->getResults()) {
        for (Operation *v : res.getUsers()) {
          // If user v exists in the destination step
          if (nodeMap.count({v, dstStep})) {
            // We assume this dataflow corresponds to the transition
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

  // Declare nodes
  for (const auto &node : nodes) {
    file << "  " << node.getDotId() << " [label=\"" << node.getLabel()
         << "\"];\n";
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
  llvm::errs().flush(); // FORCE FLUSH
}