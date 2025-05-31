//===- LSQSizingSupport.h - Support functions for LSQ Sizing ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements Helper Classes and Functions for the LSQ sizing pass.
// The Helper functions mainly consist of the CFDFCGraph class which is used
// to represent the CFDFC as an adjacency list graph and provides functions to
// find paths, calculate latencies and start times of nodes.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include <list>
#include <mlir/IR/Operation.h>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H
#define DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H

namespace dynamatic {
namespace experimental {
namespace lsqsizing {

/// Define a structure for a graph node
struct CFDFCNode {
  /// Latency of the operation
  int latency;
  /// Earliest start time of the operation
  int earliestStartTime;
  /// Pointer to the operation (nullptr if its an articificial node)
  mlir::Operation *op;
  /// Adjacency list (stores keys of adjacent nodes)
  std::set<std::string> edges;
  /// Backedge Adjacency list (stores keys of adjacent nodes connected by
  /// backedges)
  std::set<std::string> backedges;
  /// Shifting edge Adjacency list (stores keys of adjacent nodes
  /// connected by shifting edges)
  std::set<std::string> shiftingedges;
};

class CFDFCGraph {
public:
  /// Constructor for the graph, which takes a Vector of BBs, which make up a
  /// single CFDFC
  CFDFCGraph(handshake::FuncOp funcOp, llvm::SetVector<unsigned> cfdfcBBs,
             TimingDatabase timingDB, unsigned II, double targetCP);

  /// Adds the edges between the start node and the start node candidates, with
  /// their respective shifting in the start time These edges are necessary to
  /// account for the case, that the latest argument arrival time, is not from a
  /// path which starts at the start node, but from a path which is not
  /// connected to the start node
  void addShiftingEdge(mlir::Operation *src, mlir::Operation *dest,
                       int shiftingLatency);

  /// adds an Edge between src and dest
  void addEdge(mlir::Operation *src, mlir::Operation *dest);

  /// Prints the graph
  void printGraph();

  /// Prints the nodes on a path and their respecitve latencies
  void printPath(std::vector<std::string> path);

  /// Finds all paths between two nodes, given the Operation pointers
  std::vector<std::vector<std::string>>
  findPaths(mlir::Operation *startOp, mlir::Operation *endOp,
            bool ignoreBackedge = false, bool ignoreShiftingEdge = true);

  /// Finds all paths between two nodes, given the Operations unique names
  std::vector<std::vector<std::string>>
  findPaths(std::string start, std::string end, bool ignoreBackedge = false,
            bool ignoreShiftingEdge = true);

  /// Returns the latency of a path
  int getPathLatency(std::vector<std::string> path);

  /// Finds the path with the highest latency between two nodes
  int findMaxPathLatency(mlir::Operation *startOp, mlir::Operation *endOp,
                         bool ignoreBackedge = false,
                         bool ignoreShiftingEdge = true,
                         bool excludeLastNodeLatency = false);

  /// Finds the path with the lowest latency between two nodes
  int findMinPathLatency(mlir::Operation *startOp, mlir::Operation *endOp,
                         bool ignoreBackedge = false,
                         bool ignoreShiftingEdge = true);

  /// Finds the longest non-cyclic path starting from a node (ignores backedges)
  std::vector<std::string> findLongestNonCyclicPath(mlir::Operation *startOp);

  /// Returns all operations in the graph
  std::vector<mlir::Operation *> getOperations();

  /// Returns all operations in the graph with a specific operation type
  template <typename OpType>
  std::vector<mlir::Operation *> getOperationsWithOpType() {
    std::vector<mlir::Operation *> ops;
    // Iterate over all nodes and use dyn_cast to filter operations by type
    for (auto &node : nodes) {
      if (node.second.op && isa<OpType>(node.second.op)) {
        ops.push_back(node.second.op);
      }
    }
    return ops;
  }

  /// Returns all operations which are connected to a specific operation
  /// (includes backedges, but skips the artificial nodes)
  std::vector<mlir::Operation *> getConnectedOps(mlir::Operation *op);

  /// Finds the path with the highest latency between two any memory
  /// dependencies This path is the worst case II, in case of collisions
  unsigned getWorstCaseII();

  /// Updates the graph to use a different II for the latency of the backedges
  void setNewII(unsigned II);

  /// iterates over the graph, starting from startOp and finds the earliest
  /// start time for each operation according to the latencies
  void setEarliestStartTimes(mlir::Operation *startOp);

  /// returns the earliest start time of an operation
  int getEarliestStartTime(mlir::Operation *op);

  /// checks if an operation is connected to the LSQ (for checking if a
  /// load/store is for a LSQ or for a MC)
  static bool isConnectedToLSQ(mlir::Operation *op);

private:
  static constexpr const char *backedgePrefix =
      "backedge_"; // Prefix for backedges
  static constexpr const char *shiftingPrefix =
      "shifting_"; // Prefix for shifting edges

  /// Map to store the nodes by their Operations unique name
  std::unordered_map<std::string, CFDFCNode> nodes;

  /// Map to store the startTime according to each connected edge, used for
  /// finding the earliestStartTime of each node
  std::unordered_map<std::string, std::unordered_map<std::string, int>>
      edgeMinLatencies;

  /// Adds a operation with its latency to the graph as a node
  void addNode(mlir::Operation *op, int latency);

  /// Adds an artificial node on a backedge, If there is already an edge it will
  /// be removed and replaced by a backedge
  void addBackedge(mlir::Operation *src, mlir::Operation *dest, int latency);

  /// Recursive Depth-First-Search for Path finding between two nodes
  void dfsAllPaths(std::string &currentNode, std::string &end,
                   std::vector<std::string> &currentPath,
                   std::set<std::string> &visited,
                   std::vector<std::vector<std::string>> &paths,
                   bool ignoreBackedges = false,
                   bool ignoreShiftingEdge = true);

  /// Recursive Depth-First-Search for finding the longest path acyclic path to
  /// any node
  void dfsLongestAcyclicPath(const std::string &currentNode,
                             std::set<std::string> &visited,
                             std::vector<std::string> &currentPath,
                             int &maxLatency,
                             std::vector<std::string> &bestPath);

  /// Adds the edges between nodes for a result value of an operation
  void addChannelEdges(mlir::Value);

  /// Adds the backedges between nodes for a result value of an operation
  void addChannelBackedges(mlir::Value, int latency);

  /// Recursive algorithm to go trough nodes and update the earliest start time
  void setEarliestStartTimes(std::string startNode,
                             std::set<std::string> &visited);

  /// Updates the start time of a node, if the start time of the previous node
  /// is higher, handles edge cases for mux,merge and cmerge for which it needs
  /// the edgeMinLatencies map.
  bool updateStartTimeForNode(std::string node, std::string prevNode);
};

} // namespace lsqsizing
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H