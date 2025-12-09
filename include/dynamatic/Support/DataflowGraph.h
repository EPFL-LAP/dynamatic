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

#ifndef DYNAMATIC_SUPPORT_DATAFLOWGRAPH_H
#define DYNAMATIC_SUPPORT_DATAFLOWGRAPH_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "experimental/Support/StdProfiler.h"
#include "llvm/ADT/SmallVector.h"

#include <set>
#include <utility>
#include <vector>

namespace dynamatic {

/// A reconvergent path is a subgraph where multiple paths diverge from a fork
/// and reconverge at a join. This is important for latency balancing.
struct ReconvergentPath {
  size_t forkNodeId; // <- The divergence point (fork operation).
  size_t joinNodeId; // <- The convergence point (merge/mux operation).
  std::set<size_t> nodeIds; // <- All nodes on paths from fork to join.

  ReconvergentPath(size_t fork, size_t join, std::set<size_t> nodes)
      : forkNodeId(fork), joinNodeId(join), nodeIds(std::move(nodes)) {}
};

struct DataflowGraphNode {
  mlir::Operation *op; // <- The underlying operation.
  unsigned step;       // <- The transition step in which the operation exists.
  size_t id; // <- Unique id in the nodes vector to help with traversal.

  DataflowGraphNode(mlir::Operation *op, unsigned step, size_t id) : op(op), step(step), id(id) {}

  //  Debugging Helpers for GraphViz and co. //
  std::string getDotId() const;
  std::string getLabel() const;
};

enum DataflowGraphEdgeType {
  INTRA_BB, // <- Edge within the same basic block (same step).
  INTER_BB, // <- Edge between different basic blocks (step i -> step i+1).
};

struct DataflowGraphEdge {
  size_t srcId;
  size_t dstId;

  mlir::Value channel;
  DataflowGraphEdgeType type;

  DataflowGraphEdge(size_t srcId, size_t dstId, DataflowGraphEdgeType type,
                    mlir::Value channel)
      : srcId(srcId), dstId(dstId), channel(channel), type(type) {}
};

// A non-cyclic dataflow graph unrolled along a transition sequence.
// Nodes are (Operation, step) pairs. Edges are typed as intra-BB or inter-BB.
class DataflowGraph {
private:
  handshake::FuncOp funcOp; // <- The function this graph represents.

  std::vector<DataflowGraphNode> nodes;
  std::vector<DataflowGraphEdge> edges;

  // Adjacency list for easy traversal, node_id -> list of edge indices.
  std::vector<llvm::SmallVector<size_t, 4>> adjList;

  // Map (Operation*, step) -> to node id for quick lookup.
  std::map<std::pair<mlir::Operation *, unsigned>, size_t> nodeMap;

  // Map from step to basic block id.
  std::map<unsigned, unsigned> stepToBB;

  // Get the node id if it exists, otherwise create it and return the new id.
  size_t getOrAddNode(mlir::Operation *op, unsigned step);

  // Add an edge to the graph.
  void addEdge(size_t srcId, size_t dstId, DataflowGraphEdgeType type,
               mlir::Value channel);

  // A helper for runDFS()
  void dfsVisit(size_t nodeId, std::vector<bool> &visited,
                llvm::raw_ostream &os);

  // Reverse adjacency list for backward traversal (built on demand).
  std::vector<llvm::SmallVector<size_t, 4>> revAdjList;

  // Build the reverse adjacency list if not already built.
  void buildReverseAdjList();

public:
  DataflowGraph(handshake::FuncOp funcOp,
                const std::vector<dynamatic::experimental::ArchBB> &sequence);

  // Getters //
  const std::vector<DataflowGraphNode> &getNodes() const { return nodes; }
  const std::vector<DataflowGraphEdge> &getEdges() const { return edges; }
  const std::vector<llvm::SmallVector<size_t, 4>> &getAdjList() const {
    return adjList;
  }

  void runDFS();
  void dumpGraphViz(llvm::StringRef filename);

  // === Reconvergent Path Analysis === //

  /// Find all reconvergent paths in the graph.
  /// A reconvergent path is where dataflow diverges at a fork and reconverges
  /// at a join (merge/mux). Returns paths with > 2 nodes (non-trivial).
  std::vector<ReconvergentPath> findReconvergentPaths();

  /// Dump reconvergent paths to a GraphViz file for visualization.
  void dumpReconvergentPaths(const std::vector<ReconvergentPath> &paths,
                             llvm::StringRef filename);
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_DATAFLOWGRAPH_H