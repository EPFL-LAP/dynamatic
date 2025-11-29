//===- TransitionCFDFC.h - Transition-based CFDFC -----------------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares a lightweight class to represent and visualize a dataflow circuit
// unrolled along a sequence of basic block transitions.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_TRANSITIONCFDFC_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_TRANSITIONCFDFC_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/StdProfiler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dynamatic {
namespace buffer {

struct TransitionNode {
  /// The underlying operation.
  mlir::Operation *op;
  /// The step index of the transition sequence (starts at 0).
  unsigned step;
  /// Unique ID for the node in the graph (index in the nodes vector).
  size_t id;

  TransitionNode(mlir::Operation *op, unsigned step, size_t id)
      : op(op), step(step), id(id) {}

  /// Returns a unique string ID for GraphViz.
  std::string getDotId() const;
  /// Returns a label for GraphViz.
  std::string getLabel() const;
};

/// A lightweight class to represent a dataflow circuit unrolled along a
/// sequence of basic block transitions.
class TransitionCFDFC {
public:
  /// Constructs the unrolled graph from the given function and transition
  /// sequence. The sequence is treated as a path. If transitions are connected
  /// (dst of i matches src of i+1), the steps are chained, reusing the node
  /// instances.
  TransitionCFDFC(handshake::FuncOp funcOp,
                  const std::vector<dynamatic::experimental::ArchBB> &sequence);

  /// Runs a DFS traversal on the graph and prints visited nodes to stdout.
  void runDFS();

  void dumpGraphViz(llvm::StringRef filename);

private:
  /// The function being analyzed.
  handshake::FuncOp funcOp;
  /// The nodes in the graph.
  std::vector<TransitionNode> nodes;

  /// Map from step index to Basic Block ID.
  std::map<unsigned, unsigned> stepToBB;

  /// Adjacency list: node index -> list of (neighbor index, channel value).
  /// The value represents the channel (edge) carrying data.
  std::vector<std::vector<std::pair<size_t, mlir::Value>>> adjList;
  /// Mapping from (Operation *, step) -> node index.
  std::map<std::pair<mlir::Operation *, unsigned>, size_t> nodeMap;
  /// Adds a node to the graph if it doesn't exist, returns the index.
  size_t getOrAddNode(mlir::Operation *op, unsigned step);
  /// Adds an edge to the graph.
  void addEdge(size_t srcId, size_t dstId, mlir::Value channel);

  /// Helper for DFS.
  void dfsVisit(size_t u, std::vector<bool> &visited, llvm::raw_ostream &os);
};

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_TRANSITIONCFDFC_H