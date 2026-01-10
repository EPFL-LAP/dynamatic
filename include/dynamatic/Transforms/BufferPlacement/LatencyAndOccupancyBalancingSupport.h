//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//
//
// Graph-based enumeration tools to latency and occupancy balance dataflow
// circuits.
//
//===-----------------------------------------------------------------------------===//

#pragma once

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include <set>

using namespace dynamatic::experimental;

#include <cstddef>

namespace dynamatic {

using NodeIdType = size_t;
using EdgeIdType = size_t;

/// NOTE: No current implementation differentiates between intra-BB and inter-BB
/// edges. Right now, it's quite useful for visualizing the graph in GraphViz.
enum DataflowGraphEdgeType {
  INTRA_BB, // <-- Edge within the same basic block.
  INTER_BB, // <-- Edge between different basic blocks.
};

struct DataflowGraphNode {
  mlir::Operation *op; // <-- The underlying Operation.
  NodeIdType id; // <-- Unique id in the nodes vector to help with traversal.

  DataflowGraphNode(mlir::Operation *op, NodeIdType id) : op(op), id(id) {}
};

struct DataflowGraphEdge {
  NodeIdType srcId;
  NodeIdType dstId;

  mlir::Value channel;
  DataflowGraphEdgeType type;

  DataflowGraphEdge(
      NodeIdType srcId, NodeIdType dstId, mlir::Value channel,
      DataflowGraphEdgeType type = DataflowGraphEdgeType::INTRA_BB)
      : srcId(srcId), dstId(dstId), channel(channel), type(type) {}
};

/// Abstract base class for dataflow graphs used in circuit analysis &
/// optimization. It does the heavy lifting of managing the graph structure and
/// traversal. Inheriting classes need to implement type-specific methods to
/// determine what constitutes a fork and join. They then can add custom logic
/// to enumerate:
///     - Reconvergent paths from acyclic graphs
///     - Synchronizing paths from Choice-Free-Circuits (CFCs)

struct DataflowSubgraphBase {
  virtual ~DataflowSubgraphBase() = default;

  /// Virtual Methods ///

  virtual bool isForkNode(NodeIdType nodeId) const = 0;
  virtual bool isJoinNode(NodeIdType nodeId) const = 0;

  virtual std::string getNodeLabel(NodeIdType nodeId) const = 0;
  virtual std::string getNodeDotId(NodeIdType nodeId) const = 0;

  /// Getters ///

  handshake::FuncOp getFuncOp() const { return funcOp; }

  handshake::FuncOp funcOp;

  std::vector<DataflowGraphNode> nodes;
  std::vector<DataflowGraphEdge> edges;

  /// NOTE: Uses node ID to index the nodes.
  std::vector<llvm::SmallVector<EdgeIdType, 4>> adjList;
  std::vector<llvm::SmallVector<EdgeIdType, 4>> revAdjList;

  NodeIdType addNode(mlir::Operation *op) {
    NodeIdType id = nodes.size();
    nodes.emplace_back(op, id);
    adjList.emplace_back();
    revAdjList.emplace_back();
    return nodes.size() - 1;
  }

  void addEdge(NodeIdType srcId, NodeIdType dstId, mlir::Value channel,
               DataflowGraphEdgeType type = DataflowGraphEdgeType::INTRA_BB) {
    edges.emplace_back(srcId, dstId, channel, type);
    adjList[srcId].push_back(edges.size() - 1);
    revAdjList[dstId].push_back(edges.size() - 1);
  }
};

/// A reconvergent path is a subgraph where multiple paths diverge from a fork
/// and reconverge at a join. This is important for latency balancing.
struct ReconvergentPath {
  NodeIdType forkNodeId;        // The divergence point
  NodeIdType joinNodeId;        // The convergence point
  std::set<NodeIdType> nodeIds; // All nodes on paths from fork to join.

  ReconvergentPath(NodeIdType fork, NodeIdType join, std::set<NodeIdType> nodes)
      : forkNodeId(fork), joinNodeId(join), nodeIds(std::move(nodes)) {}
};

/// Enumerate all possible sequences of the given length from the transitions.
/// Treats transitions as edges in a BB graph and finds all paths of length N
/// where each transition's dstBB matches the next transition's srcBB.
/// Example:
/// 1 -> 2
/// 2 -> 3
/// 1 -> 1 (self-loop)
/// enumerateTransitionSequences(transitions, 3);
/// Output: [1, 2, 3], [1, 1, 2], [1, 1, 1]
inline std::vector<std::vector<ArchBB>>
enumerateTransitionSequences(llvm::ArrayRef<ArchBB> transitions,
                             size_t sequenceLength) {
  // 'sequenceLength' is the number of steps to visit.
  // Number of transitions needed = sequenceLength - 1.
  // Minimum is 2 steps (1 transition).
  if (sequenceLength < 2 || transitions.empty()) {
    return {};
  }

  unsigned numTransitions = sequenceLength - 1;

  // Maps BB -> list of transitions starting from that BB.
  std::map<unsigned, std::vector<const ArchBB *>> adjList;
  for (const auto &t : transitions) {
    adjList[t.srcBB].push_back(&t);
  }

  std::vector<std::vector<ArchBB>> result;

  // enumerate all sequences with the required number of transitions using DFS
  std::function<void(std::vector<experimental::ArchBB> &)> dfs =
      [&](std::vector<experimental::ArchBB> &current) {
        if (current.size() == numTransitions) {
          result.push_back(current);
          return;
        }

        unsigned currentBB = current.back().dstBB;
        auto nextTransitionsIt = adjList.find(currentBB);
        if (nextTransitionsIt == adjList.end())
          return;

        for (const auto *nextTransition : nextTransitionsIt->second) {
          current.push_back(*nextTransition);
          dfs(current);
          current.pop_back();
        }
      };

  // Start from each transition
  for (const auto &t : transitions) {
    std::vector<experimental::ArchBB> seq = {t};
    dfs(seq);
  }

  return result;
};

/// A dataflow graph specialized for reconvergent path analysis.
/// IMPORTANT: This class assumes the graph an ACYCLIC transition sequence.
class ReconvergentPathFinderGraph : public DataflowSubgraphBase {
public:
  bool isForkNode(NodeIdType nodeId) const override {
    return isa<handshake::ForkOp, handshake::LazyForkOp,
               handshake::EagerForkLikeOpInterface>(nodes[nodeId].op);
  }

  // The only nodes with two inputs that allow for both
  // inputs to be active at the same time. Unlike: ControlMergeOp and MergeOp.
  /// NOTE: When it belongs to a CFDFC, MuxOp behaves like a join node.
  bool isJoinNode(NodeIdType nodeId) const override {
    return isa<handshake::MuxOp, handshake::JoinLikeOpInterface,
               handshake::ConditionalBranchOp>(nodes[nodeId].op);
  }

  std::string getNodeLabel(NodeIdType nodeId) const override;
  std::string getNodeDotId(NodeIdType nodeId) const override;

  std::vector<ReconvergentPath> findReconvergentPaths() const;

  /// Build the graph from a given transition sequence.
  void buildGraphFromSequence(handshake::FuncOp funcOp,
                              llvm::ArrayRef<ArchBB> sequence);

  /// Within the transition sequence, we may have transitions that look like
  /// Step 0: BB1 -> Step 1: BB1 -> Step 2: BB2. Steps are the way to
  /// distinguish between specific operations of the same BB accross different
  /// transitions.
  unsigned getNodeStep(NodeIdType nodeId) const {
    return nodeIdToStep.at(nodeId);
  };

  /// Get the BB id for a given step. Returns -1 if step not found.
  unsigned getStepBB(unsigned step) const {
    auto it = stepToBB.find(step);
    return it != stepToBB.end() ? it->second : -1;
  }

  // Debugging Methods //

  void dumpReconvergentPaths(llvm::ArrayRef<ReconvergentPath> paths,
                             llvm::StringRef filename) const;

  void dumpTransitionGraph(llvm::StringRef filename) const;

  /// Dump multiple graphs to a single GraphViz file.
  /// Each graph is placed in its own cluster subgraph.
  static void dumpAllGraphs(llvm::ArrayRef<ReconvergentPathFinderGraph> graphs,
                            llvm::StringRef filename);

  /// Dump all reconvergent paths from multiple graphs to a single GraphViz
  /// file. Each path is placed in its own cluster subgraph with a graph index
  /// prefix. The input is a vector of GraphPathsForDumping objects. Each object
  /// contains:
  /// - graph: Pointer to the ReconvergentPathFinderGraph for this sequence.
  /// - paths: Vector of ReconvergentPath objects for this sequence.

  struct GraphPathsForDumping {
    const ReconvergentPathFinderGraph *graph;
    std::vector<ReconvergentPath> paths;
  };

  static void
  dumpAllReconvergentPaths(llvm::ArrayRef<GraphPathsForDumping> graphPaths,
                           llvm::StringRef filename);

private:
  std::map<unsigned, unsigned> stepToBB;

  /// Maps node ID to it's step number.
  std::map<unsigned, unsigned> nodeIdToStep;

  /// Maps (Operation*, step) to node ID for O(1) lookup.
  std::map<std::pair<mlir::Operation *, unsigned>, NodeIdType> nodeMap;

  /// Get the node ID for an operation at a given step, creating it if needed.
  NodeIdType getOrAddNode(mlir::Operation *op, unsigned step) {
    auto key = std::make_pair(op, step);
    if (auto it = nodeMap.find(key); it != nodeMap.end())
      return it->second;

    NodeIdType id = addNode(op);
    nodeMap[key] = id;
    nodeIdToStep[id] = step;
    return id;
  }
};

struct SimpleCycle {
  llvm::SmallVector<NodeIdType> nodes; // <-- The node IDs of the cycle.
  SimpleCycle(llvm::ArrayRef<NodeIdType> nodes) : nodes(nodes) {}

  /// Check if this cycle shares any nodes with another cycle.
  bool isDisjointFrom(const SimpleCycle &other) const;
};

struct EdgesToJoin {
  NodeIdType joinId;

  /// Edge indices (into nonCyclicAdjList) on any path from cycle to join.
  std::vector<EdgeIdType> edgesFromCycleOne;
  std::vector<EdgeIdType> edgesFromCycleTwo;

  EdgesToJoin(NodeIdType join) : joinId(join) {}
};

struct SynchronizingCyclePair {
  SimpleCycle cycleOne;
  SimpleCycle cycleTwo;

  std::vector<EdgesToJoin> edgesToJoins;

  SynchronizingCyclePair(SimpleCycle one, SimpleCycle two,
                         std::vector<EdgesToJoin> edges)
      : cycleOne(std::move(one)), cycleTwo(std::move(two)),
        edgesToJoins(std::move(edges)) {}
};

class SynchronizingCyclesFinderGraph : public DataflowSubgraphBase {
public:
  /// Build the graph from a CFDFC.
  void buildFromCFDFC(handshake::FuncOp funcOp, const buffer::CFDFC &cfdfc);

  /// Find all simple cycles in the graph.
  std::vector<SimpleCycle> findAllCycles() const;

  /// Find all pairs of synchronizing cylces.
  std::vector<SynchronizingCyclePair> findSynchronizingCyclePairs();

  bool isForkNode(NodeIdType nodeId) const override {
    return isa<handshake::ForkOp, handshake::LazyForkOp,
               handshake::EagerForkLikeOpInterface>(nodes[nodeId].op);
  }

  /// NOTE: When it belongs to a CFDFC, MuxOp behaves like a join node.
  bool isJoinNode(NodeIdType nodeId) const override {
    return isa<handshake::MuxOp, handshake::JoinLikeOpInterface,
               handshake::ConditionalBranchOp>(nodes[nodeId].op);
  }

  std::string getNodeLabel(NodeIdType nodeId) const override;
  std::string getNodeDotId(NodeIdType nodeId) const override;

  /// Dump a single synchronizing cycle pair to a GraphViz file.
  void dumpSynchronizingCyclePair(const SynchronizingCyclePair &pair,
                                  llvm::StringRef filename) const;

  /// Dump all synchronizing cycle pairs to a single GraphViz file.
  void
  dumpAllSynchronizingCyclePairs(llvm::ArrayRef<SynchronizingCyclePair> pairs,
                                 llvm::StringRef filename) const;

private:
  std::map<mlir::Operation *, NodeIdType> opToNodeId;

  /// Adjacency list for the non-cyclic subgraph (stores edge indices).
  std::vector<std::vector<EdgeIdType>> nonCyclicAdjList;

  NodeIdType getOrAddNode(mlir::Operation *op);

  void computeSccsAndBuildNonCyclicSubgraph();

  /// Find all edges (from nonCyclicAdjList) on any path from cycle to join.
  /// An edge is included if its source is reachable from the cycle and its
  /// destination can reach the join.
  /// @returns A vector of indices into the edges vector.
  std::vector<EdgeIdType> findEdgesToJoin(const SimpleCycle &cycle,
                                      NodeIdType joinId) const;

  /// Get all join node IDs in the graph.
  std::vector<NodeIdType> getAllJoins() const;
};

// Helper struct needed for Boost's tiernan_all_cycles algorithm.
// As opposed to just returning cycles directly, it calls a method on a
// user-provided visitor object each time a cycle is found.
struct CycleCollector {
  std::vector<SimpleCycle> &cycles;

  template <typename Path, typename Graph>
  void cycle(const Path &p, const Graph &) {
    llvm::SmallVector<NodeIdType> nodeIds;
    for (auto v : p) {
      nodeIds.push_back(v);
    }
    cycles.emplace_back(std::move(nodeIds));
  }
};

} // namespace dynamatic