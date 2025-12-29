//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class for finding pairs of synchronizing cycles in a dataflow graph.
// Given a dataflow graph, find pairs of cycles that are synchronizing, 
// according to the definition found in Section 4 of this paper:
//
// [Xu, Josipović, FPGA'24 (https://dl.acm.org/doi/10.1145/3626202.36375)] 
//
// Definition 3: Two cycles are a pair of synchronizing cycles in a
// dataflow circuit if the following properties hold: (1) The two cycles
// are disjoint (i.e. they do not have any common units) and belong to
// the same CFC (defined in Section 3.2). (2) There exists at least one
// join that is reachable from both cycles without crossing any edge
// on the cycle in the CFC they belong to.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_DATAFLOWGRAPH_SYNCHRONIZINGCYCLESFINDER_H
#define DYNAMATIC_SUPPORT_DATAFLOWGRAPH_SYNCHRONIZINGCYCLESFINDER_H

#include "dynamatic/Support/DataflowGraph/DataflowGraphBase.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>

namespace dynamatic {

struct SimpleCycle {
  std::vector<size_t> nodes; // <-- The node IDs of the cycle.
  SimpleCycle(std::vector<size_t> nodes) : nodes(std::move(nodes)) {}

  /// Check if this cycle shares any nodes with another cycle.
  bool isDisjointFrom(const SimpleCycle &other) const;
};

struct PathToJoin {
  size_t joinId;

  std::vector<size_t> pathFromCycleOne;
  std::vector<size_t> pathFromCycleTwo;

  PathToJoin(size_t join) : joinId(join) {}
};

struct SynchronizingCyclePair {
  const SimpleCycle cycleOne;
  const SimpleCycle cycleTwo;

  std::vector<PathToJoin> pathsToJoins;

  SynchronizingCyclePair(SimpleCycle one, SimpleCycle two,
                         std::vector<PathToJoin> paths)
      : cycleOne(std::move(one)), cycleTwo(std::move(two)),
        pathsToJoins(std::move(paths)) {}
};

class SynchronizingCyclesFinderGraph
: public DataflowGraphBase<mlir::Operation *, mlir::Value>{
public:
  /// Build the graph from a CFDFC.
  void buildFromCFDFC(handshake::FuncOp funcOp, const buffer::CFDFC &cfdfc);
  
  /// Find all simple cycles in the graph.
  std::vector<SimpleCycle> findAllCycles() const;

  /// Find all pairs of synchronizing cylces.
  std::vector<SynchronizingCyclePair> findSynchronizingCyclePairs();

  bool isForkNode(size_t nodeId) const override;
  bool isJoinNode(size_t nodeId) const override;

  std::string getNodeLabel(size_t nodeId) const override;
  std::string getNodeDotId(size_t nodeId) const override;

  /// Dump a single synchronizing cycle pair to a GraphViz file.
  void dumpSynchronizingCyclePair(const SynchronizingCyclePair &pair,
                                  llvm::StringRef filename) const;

  /// Dump all synchronizing cycle pairs to a single GraphViz file.
  void dumpAllSynchronizingCyclePairs(
      const std::vector<SynchronizingCyclePair> &pairs,
      llvm::StringRef filename) const;

private:
  std::map<mlir::Operation *, size_t> opToNodeId;

  /// SCC ID for each node.
  std::vector<size_t> nodeSccId;

  /// Adjacency list for the non-cyclic subgraph.
  std::vector<std::vector<size_t>> nonCyclicAdjList;

  size_t getOrAddNode(mlir::Operation *op);

  void computeSccsAndBuildNonCyclicSubgraph();

  /// Find path from a cycle to a join using BFS on non-cyclic subgraph.
  std::vector<size_t> findPathToJoin(const SimpleCycle &cycle, size_t joinId) const;

  /// Get all join node IDs in the graph.
  std::vector<size_t> getAllJoins() const;
};

struct CycleCollector {
  std::vector<SimpleCycle> &cycles;

  template <typename Path, typename Graph>
  void cycle(const Path &p, const Graph &) {
    std::vector<size_t> nodeIds;
    for (auto v : p) {
      nodeIds.push_back(v);
    }
    cycles.emplace_back(std::move(nodeIds));
  }
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_DATAFLOWGRAPH_SYNCHRONIZINGCYCLESFINDER_H
