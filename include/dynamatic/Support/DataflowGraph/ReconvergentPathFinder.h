//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class for finding reconvergent paths in a dataflow graph.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_DATAFLOWGRAPH_RECONVERGENTPATHFINDER_H
#define DYNAMATIC_SUPPORT_DATAFLOWGRAPH_RECONVERGENTPATHFINDER_H

#include "dynamatic/Support/DataflowGraph/DataflowGraphBase.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/IR/Operation.h"
#include <set>

using namespace dynamatic::experimental;

namespace dynamatic {

/// A reconvergent path is a subgraph where multiple paths diverge from a fork
/// and reconverge at a join. This is important for latency balancing.
struct ReconvergentPath {
  size_t forkNodeId;        // The divergence point
  size_t joinNodeId;        // The convergence point
  std::set<size_t> nodeIds; // All nodes on paths from fork to join.

  ReconvergentPath(size_t fork, size_t join, std::set<size_t> nodes)
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
static std::vector<std::vector<ArchBB>>
enumerateTransitionSequences(const std::vector<ArchBB> &transitions,
                             size_t sequenceLength) {
  // 'sequenceLength' is the number of steps to visit.
  // Number of transitions needed = sequenceLength - 1.
  // Minimum is 2 steps (1 transition).
  if (sequenceLength < 2 || transitions.empty()) {
    return {};
  }

  unsigned numTransitions = sequenceLength - 1;

  // Maps BB -> list of transitions starting from that BB.
  std::map<unsigned, std::vector<const ArchBB *>> transitionsFrom;
  for (const auto &t : transitions) {
    transitionsFrom[t.srcBB].push_back(&t);
  }

  std::vector<std::vector<ArchBB>> result;

  // enumerate all sequences with the required number of transitions using DFS
  std::function<void(std::vector<experimental::ArchBB> &)> dfs =
      [&](std::vector<experimental::ArchBB> &current) {
        if (current.size() == numTransitions) {
          result.push_back(current);
          return;
        }

        unsigned lastDstBB = current.back().dstBB;
        auto it = transitionsFrom.find(lastDstBB);
        if (it == transitionsFrom.end())
          return;

        for (const auto *next : it->second) {
          current.push_back(*next);
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
class ReconvergentPathFinderGraph
    : public DataflowGraphBase<mlir::Operation *, mlir::Value> {
public:
  bool isForkNode(size_t nodeId) const override;
  bool isJoinNode(size_t nodeId) const override;

  std::string getNodeLabel(size_t nodeId) const override;
  std::string getNodeDotId(size_t nodeId) const override;

  std::vector<ReconvergentPath> findReconvergentPaths() const;

  /// Build the graph from a given transition sequence.
  void buildGraphFromSequence(handshake::FuncOp funcOp,
                              const std::vector<ArchBB> &sequence);

  /// Within the transition sequence, we may have transitions that look like
  /// Step 0: BB1 -> Step 1: BB1 -> Step 2: BB2. Steps are the way to
  /// distinguish between specific operations of the same BB accross different
  /// transitions.
  unsigned getNodeStep(size_t nodeId) const { return nodeIdToStep.at(nodeId); };

  /// Get the BB id for a given step. Returns -1 if step not found.
  unsigned getStepBB(unsigned step) const {
    auto it = stepToBB.find(step);
    return it != stepToBB.end() ? it->second : -1;
  }

  // Debugging Methods //

  void dumpReconvergentPaths(const std::vector<ReconvergentPath> &paths,
                             llvm::StringRef filename) const;

  void dumpTransitionGraph(llvm::StringRef filename) const;

  /// Dump multiple graphs to a single GraphViz file.
  /// Each graph is placed in its own cluster subgraph.
  static void
  dumpAllGraphs(const std::vector<ReconvergentPathFinderGraph> &graphs,
                llvm::StringRef filename);

  /// Dump all reconvergent paths from multiple graphs to a single GraphViz
  /// file. Each path is placed in its own cluster subgraph with a graph index
  /// prefix. The input is a vector of (sequenceIndex, (graph, paths)) pairs.
  static void dumpAllReconvergentPaths(
      const std::vector<
          std::pair<size_t, std::pair<const ReconvergentPathFinderGraph *,
                                      std::vector<ReconvergentPath>>>>
          &graphPaths,
      llvm::StringRef filename);

private:
  std::map<unsigned, unsigned> stepToBB;

  /// Maps node ID to it's step number.
  std::map<unsigned, unsigned> nodeIdToStep;

  /// Maps (Operation*, step) to node ID for O(1) lookup.
  std::map<std::pair<mlir::Operation *, unsigned>, size_t> nodeMap;

  /// Get the node ID for an operation at a given step, creating it if needed.
  size_t getOrAddNode(mlir::Operation *op, unsigned step) {
    auto key = std::make_pair(op, step);
    if (auto it = nodeMap.find(key); it != nodeMap.end())
      return it->second;

    size_t id = addNode(op);
    nodeMap[key] = id;
    nodeIdToStep[id] = step;
    return id;
  }
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_DATAFLOWGRAPH_RECONVERGENTPATHFINDER_H