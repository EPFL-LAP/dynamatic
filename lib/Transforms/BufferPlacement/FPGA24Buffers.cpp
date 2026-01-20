//===- FPGA24Buffers.cpp - FPGA'24 buffer placement ----------------------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of FPGA'24 Latency and Occupancy Balancing buffer placement.
// Based on: [Xu, Josipović, FPGA'24]
// (https://dl.acm.org/doi/10.1145/3626202.3637570) I will be referencing the
// paper by including the relevant equations and definitions like so: (Paper:
// ...). Please note that a summary of the LP's is provided in the `.h`.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/FPGA24Buffers.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/LatencyAndOccupancyBalancingSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <list>
#include <set>
#include <string>

using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpga24;

#define DEBUG_TYPE "fpga24-buffers"

/// Constants ///

/// Big-M constant for imbalance constraints.
// (Paper: Section 4, Equation 2)
static constexpr double BIG_M = 1000.0;

/// Weight for stall penalty vs latency cost (>> LATENCY_WEIGHT to prioritize
/// stalls). (Paper: TODO)
static constexpr double LATENCY_WEIGHT = 1.0;
static constexpr double STALL_WEIGHT = 1000.0;

/// Upper bound for occupancy
static constexpr double MAX_OCCUPANCY = 100.0;

/// Helper Functions ///

/// Get the latency D_u of a unit.
// (Paper: Section 4, Table 1)
static double getUnitLatency(Operation *unit, const TimingDatabase &timingDB,
                             double targetPeriod) {
  double latency = 0.0;
  if (failed(
          timingDB.getLatency(unit, SignalType::DATA, latency, targetPeriod))) {
    return 0.0;
  }

  return latency;
}

/// Get the bitwidth B_c of a channel.
/// (Paper: Section 4, Table 1)
static unsigned getChannelBitwidth(Value channel) {
  return handshake::getHandshakeTypeBitWidth(channel.getType());
}

/// Check if a unit has variable latency.
/// In our case, this is the case for LSQ-connected loads/stores.
static bool hasVariableLatency(Operation *unit) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(unit)) {
    auto memOp = findMemInterface(loadOp.getAddress());
    if (isa_and_present<handshake::LSQOp>(memOp))
      return true;
  }

  if (auto storeOp = dyn_cast<handshake::StoreOp>(unit)) {
    auto memOp = findMemInterface(storeOp.getAddress());
    if (isa_and_present<handshake::LSQOp>(memOp))
      return true;
  }

  return false;
}

/// Seeing if a path contains any unit with variable latency.
static bool checkForVariableLatency(const SmallVector<NodeIdType> &nodeIds,
                                    const DataflowSubgraphBase &graph) {
  return std::any_of(nodeIds.begin(), nodeIds.end(), [&](NodeIdType nodeId) {
    Operation *op = graph.nodes[nodeId].op;
    return hasVariableLatency(op);
  });
}

/// Structure to hold a simple path as a sequence of edges.
struct SimplePath {
  SmallVector<EdgeIdType> edges;
  SmallVector<NodeIdType> nodes;
};

/// Enumerate all simple paths from startNode to endNode in the given subgraph.
/// Only considers nodes in the allowedNodes set.
static void enumerateSimplePaths(const DataflowSubgraphBase &graph,
                                 NodeIdType startNode, NodeIdType endNode,
                                 const std::set<NodeIdType> &allowedNodes,
                                 std::vector<SimplePath> &outPaths) {
  std::vector<bool> visited(graph.nodes.size(), false);
  SimplePath currentPath;

  std::function<void(NodeIdType)> dfs = [&](NodeIdType current) {
    if (current == endNode) {
      outPaths.push_back(currentPath);
      return;
    }

    visited[current] = true;

    for (EdgeIdType edgeId : graph.adjList[current]) {
      const auto &edge = graph.edges[edgeId];
      NodeIdType next = edge.dstId;

      if (!visited[next] && allowedNodes.count(next)) {
        currentPath.edges.push_back(edgeId);
        currentPath.nodes.push_back(next);
        dfs(next);
        currentPath.edges.pop_back();
        currentPath.nodes.pop_back();
      }
    }

    visited[current] = false;
  };

  currentPath.nodes.push_back(startNode);
  dfs(startNode);
}

/// LatencyBalancingMILP Implementation ///

LatencyBalancingMILP::LatencyBalancingMILP(
    CPSolver::SolverKind solverKind, int timeout, FuncInfo &funcInfo,
    const TimingDatabase &timingDB, double targetPeriod,
    ArrayRef<ReconvergentPathWithGraph> reconvergentPaths,
    ArrayRef<SynchronizingCyclePair> syncCyclePairs,
    const SynchronizingCyclesFinderGraph &syncGraph, ArrayRef<CFDFC *> cfdfcs)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB,
                          targetPeriod),
      reconvergentPaths(reconvergentPaths), syncCyclePairs(syncCyclePairs),
      syncGraph(syncGraph), cfdfcs(cfdfcs) {
  setup();
}

void LatencyBalancingMILP::setup() {
  if (unsatisfiable)
    return;

  LLVM_DEBUG(llvm::errs() << "[LP1] Adding latency variables...\n");
  addLatencyVariables();
  LLVM_DEBUG(llvm::errs() << "[LP1] Adding reconvergent path constraints ("
                          << reconvergentPaths.size() << " paths)...\n");
  addReconvergentPathConstraints();
  LLVM_DEBUG(llvm::errs() << "[LP1] Adding sync cycle constraints ("
                          << syncCyclePairs.size() << " pairs)...\n");
  addSyncCycleConstraints();
  LLVM_DEBUG(llvm::errs() << "[LP1] Adding stall propagation constraints...\n");
  addStallPropagationConstraints();
  LLVM_DEBUG(llvm::errs() << "[LP1] Adding cycle time constraints...\n");
  addCycleTimeConstraints();
  LLVM_DEBUG(llvm::errs() << "[LP1] Setting objective...\n");
  setLatencyBalancingObjective();
  markReadyToOptimize();
  LLVM_DEBUG(llvm::errs() << "[LP1] Setup complete.\n");
}

/// The latency variable L_c is the number of extra latencies to be added to a
/// channel. It will be used in the input of the occupancy balancing LP. Defined
/// in (Paper: Section 4, Table 1).
void LatencyBalancingMILP::addLatencyVariables() {
  /// Collect all channels that need L_c variables:
  /// 1. Channels in synchronization patterns (for balancing).
  /// 2. ALL channels in CFDFCs (for cycle time constraints).
  /// Relevant: (Paper: Section 4, Equation 1, 5).
  DenseSet<Value> allChannels;

  /// From reconvergent paths:
  for (const auto &pathWithGraph : reconvergentPaths) {
    const ReconvergentPath &path = pathWithGraph.path;
    const ReconvergentPathFinderGraph *graph = pathWithGraph.graph;
    for (NodeIdType nodeId : path.nodeIds) {
      for (EdgeIdType edgeId : graph->adjList[nodeId]) {
        const auto &edge = graph->edges[edgeId];
        if (path.nodeIds.count(edge.dstId)) {
          allChannels.insert(edge.channel);
        }
      }
    }
  }

  /// From synchronizing cycle pairs:
  for (const auto &pair : syncCyclePairs) {
    /// Edges in cycle one
    for (size_t i = 0; i < pair.cycleOne.nodes.size(); ++i) {
      NodeIdType src = pair.cycleOne.nodes[i];
      NodeIdType dst =
          pair.cycleOne.nodes[(i + 1) % pair.cycleOne.nodes.size()];
      for (EdgeIdType edgeId : syncGraph.adjList[src]) {
        if (syncGraph.edges[edgeId].dstId == dst) {
          allChannels.insert(syncGraph.edges[edgeId].channel);
        }
      }
    }
    /// Edges in cycle two
    for (size_t i = 0; i < pair.cycleTwo.nodes.size(); ++i) {
      NodeIdType src = pair.cycleTwo.nodes[i];
      NodeIdType dst =
          pair.cycleTwo.nodes[(i + 1) % pair.cycleTwo.nodes.size()];
      for (EdgeIdType edgeId : syncGraph.adjList[src]) {
        if (syncGraph.edges[edgeId].dstId == dst) {
          allChannels.insert(syncGraph.edges[edgeId].channel);
        }
      }
    }
    /// Edges to joins
    for (const auto &edgeInfo : pair.edgesToJoins) {
      for (EdgeIdType edgeId : edgeInfo.edgesFromCycleOne) {
        allChannels.insert(syncGraph.edges[edgeId].channel);
      }
      for (EdgeIdType edgeId : edgeInfo.edgesFromCycleTwo) {
        allChannels.insert(syncGraph.edges[edgeId].channel);
      }
    }
  }

  /// Also include ALL channels from CFDFCs for cycle time constraints
  /// This ensures we can add latency to any channel to satisfy the following
  /// equations: (Paper: Section 4, Equation 5; Section 7, Equation 15)
  for (CFDFC *cfdfc : cfdfcs) {
    for (Value channel : cfdfc->channels) {
      allChannels.insert(channel);
    }
  }

  LLVM_DEBUG(llvm::errs() << "[LP1]   Found " << allChannels.size()
                          << " channels (patterns + CFDFCs)\n");

  /// Create variables for each channel:
  for (Value channel : allChannels) {
    std::string name = getUniqueName(*channel.getUses().begin());
    ChannelVars &chVars = vars.channelVars[channel];

    /// L_c: extra latency to add to the channel for balancing (integer >= 0).
    /// (Paper: Section 4, Table 1)
    chVars.extraLatency =
        model->addVar("L_" + name, CPVar::INTEGER, 0, std::nullopt);

    /// S_c: whether the channel is stalled due to imbalance (binary).
    /// (Paper: Section 4, Table 1)
    chVars.stalled = model->addVar("S_" + name, CPVar::BOOLEAN, 0, 1);

    /// R_c: whether the channel has L > 0, i.e., channel cut (binary).
    /// (Paper: Section 4, Table 1)
    chVars.bufPresent = model->addVar("R_" + name, CPVar::BOOLEAN, 0, 1);
  }

  LLVM_DEBUG(llvm::errs() << "[LP1]   Created " << vars.channelVars.size()
                          << " channel variables\n");

  /// Add R_c constraints,to link the binary R_c to integer L_c.
  /// (Paper: Section 4, Equation 6)
  for (auto &[channel, chVars] : vars.channelVars) {
    std::string name = getUniqueName(*channel.getUses().begin());
    /// L_c >= R_c (if R_c=1, then L_c >= 1)
    model->addConstr(chVars.extraLatency >= chVars.bufPresent,
                     "R_lower_" + name);
    /// M*R_c >= L_c (if L_c > 0, then R_c must be 1)
    model->addConstr(BIG_M * chVars.bufPresent >= chVars.extraLatency,
                     "R_upper_" + name);
  }

  /// Create pattern imbalance variables for reconvergent paths
  vars.reconvergentPathVars.resize(reconvergentPaths.size());
  for (size_t i = 0; i < reconvergentPaths.size(); ++i) {
    vars.reconvergentPathVars[i].imbalanced =
        model->addVar("s_rp_" + std::to_string(i), CPVar::BOOLEAN, 0, 1);
  }

  /// Create pattern imbalance variables for synchronizing cycles
  vars.syncCycleVars.resize(syncCyclePairs.size());
  for (size_t i = 0; i < syncCyclePairs.size(); ++i) {
    vars.syncCycleVars[i].imbalanced =
        model->addVar("s_sc_" + std::to_string(i), CPVar::BOOLEAN, 0, 1);
  }

  LLVM_DEBUG(llvm::errs() << "[LP1]   Created " << reconvergentPaths.size()
                          << " reconvergent path vars, "
                          << syncCyclePairs.size() << " sync cycle vars\n");
}

void LatencyBalancingMILP::addReconvergentPathConstraints() {
  size_t totalPaths = reconvergentPaths.size();
  for (size_t pathIdx = 0; pathIdx < totalPaths; ++pathIdx) {
    if (pathIdx % 10 == 0 || pathIdx == totalPaths - 1) {
      LLVM_DEBUG(llvm::errs() << "[LP1]   Processing reconvergent path "
                              << pathIdx + 1 << "/" << totalPaths << "\n");
    }

    const ReconvergentPathWithGraph &pathWithGraph = reconvergentPaths[pathIdx];
    const ReconvergentPath &path = pathWithGraph.path;
    const ReconvergentPathFinderGraph *graph = pathWithGraph.graph;
    CPVar &patternImbalanced = vars.reconvergentPathVars[pathIdx].imbalanced;

    /// Check if any unit in the path has variable latency.
    bool hasVarLatency = false;
    for (NodeIdType nodeId : path.nodeIds) {
      Operation *op = graph->nodes[nodeId].op;
      if (hasVariableLatency(op)) {
        hasVarLatency = true;
        break;
      }
    }

    /// If variable latency exists, there is no opportunity to balance the
    /// pattern that contains it. Thus, we force the pattern to be marked
    /// imbalanced. (Paper: Section 4, Equation 3)
    if (hasVarLatency) {
      model->addConstr(patternImbalanced == 1,
                       "varLatency_rp_" + std::to_string(pathIdx));
      continue;
    }

    /// Enumerate all simple paths from fork to join
    NodeIdType forkId = path.forkNodeId;
    NodeIdType joinId = path.joinNodeId;

    std::vector<SimplePath> allPaths;
    enumerateSimplePaths(*graph, forkId, joinId, path.nodeIds, allPaths);

    LLVM_DEBUG(llvm::errs()
               << "[LP1]     -> " << allPaths.size() << " simple paths\n");

    /// TODO(ziad): Evaluate if limiting the number of simple paths is really
    /// necessary.
    // constexpr size_t maxSimplePaths = 100;
    // if (allPaths.size() > maxSimplePaths) {
    //   LLVM_DEBUG(llvm::errs()
    //              << "[LP1]     WARNING: Too many paths, marking as
    //              imbalanced\n");
    //   model->addConstr(patternImbalanced == 1,
    //                    "tooManyPaths_rp_" + std::to_string(pathIdx));
    //   continue;
    // }

    // if (allPaths.size() < 2) {
    //   /// Not actually reconvergent, no balancing needed
    //   model->addConstr(patternImbalanced == 0,
    //                    "notReconvergent_rp_" + std::to_string(pathIdx));
    //   continue;
    // }

    /// Build latency expressions for each path.
    /// Latency(p) = sum(D_u for units u in p) + sum(L_c for channels c in p)
    /// where D_u is unit latency (constant) and L_c is extra latency (variable)
    /// (Paper: Section 4, Equation 1)
    std::vector<LinExpr> pathLatencies;
    std::vector<double> pathBaseLatencies; /// For debugging
    std::vector<unsigned> pathNumVars;     /// Number of L_c vars per path
    std::vector<unsigned> pathNumEdges;    /// Total edges per path
    for (const auto &simplePath : allPaths) {
      LinExpr pathLatency;
      double baseLatency = 0.0;
      unsigned numVars = 0;

      /// Add unit latencies D_u along the path (constants)
      for (NodeIdType nodeId : simplePath.nodes) {
        Operation *op = graph->nodes[nodeId].op;
        double unitLat = getUnitLatency(op, timingDB, targetPeriod);
        pathLatency += unitLat;
        baseLatency += unitLat;
      }

      /// Add extra latency variables L_c for channels (decision variables)
      for (EdgeIdType edgeId : simplePath.edges) {
        Value channel = graph->edges[edgeId].channel;
        if (vars.channelVars.count(channel)) {
          pathLatency += vars.channelVars[channel].extraLatency;
          numVars++;
        }
      }

      pathLatencies.push_back(pathLatency);
      pathBaseLatencies.push_back(baseLatency);
      pathNumVars.push_back(numVars);
      pathNumEdges.push_back(simplePath.edges.size());
    }

    /// Debug: Log path latencies for this reconvergent pattern (only for first
    /// few patterns with different base latencies)
    /// TODO(ziad): Remove this after making sure the LP is working correctly.
    static int debugPathCount = 0;
    if (pathBaseLatencies.size() >= 2) {
      double minLat =
          *std::min_element(pathBaseLatencies.begin(), pathBaseLatencies.end());
      double maxLat =
          *std::max_element(pathBaseLatencies.begin(), pathBaseLatencies.end());
      if (maxLat - minLat > 0.5) {
        LLVM_DEBUG(llvm::errs()
                   << "[LP1] Reconvergent path " << pathIdx
                   << ": base latencies differ by " << (maxLat - minLat)
                   << " (min=" << minLat << ", max=" << maxLat << ")\n");
        /// Show details of each path
        for (size_t i = 0; i < pathBaseLatencies.size(); i++) {
          LLVM_DEBUG(llvm::errs()
                     << "  Path " << i << ": base=" << pathBaseLatencies[i]
                     << ", edges=" << pathNumEdges[i]
                     << ", L_c vars=" << pathNumVars[i] << "\n");
        }
        /// For the first pattern with difference, show channel names on short
        /// vs long paths
        if (debugPathCount < 1) {
          debugPathCount++;
          LLVM_DEBUG(llvm::errs() << "  [DETAIL] Channels on each path:\n");
          for (size_t i = 0; i < allPaths.size() && i < 3; i++) {
            LLVM_DEBUG(llvm::errs() << "    Path " << i << " (base="
                                    << pathBaseLatencies[i] << "):\n");
            for (EdgeIdType edgeId : allPaths[i].edges) {
              Value channel = graph->edges[edgeId].channel;
              std::string name =
                  std::string(getUniqueName(*channel.getUses().begin()));
              bool hasVar = vars.channelVars.count(channel) > 0;
              LLVM_DEBUG(llvm::errs()
                         << "      " << name
                         << (hasVar ? " [HAS L_c]" : " [NO L_c]") << "\n");
            }
          }
        }
      }
    }

    /// Add imbalance constraints for each pair of paths:
    /// |L_i - L_j| <= M * s_p.
    /// Which translates to:
    ///   L_i - L_j <= M * s_p.
    ///   L_j - L_i <= M * s_p.
    /// (Paper: Section 4, Equation 2)
    if (pathBaseLatencies.empty())
      continue;

    for (size_t i = 0; i < pathLatencies.size(); ++i) {
      for (size_t j = i + 1; j < pathLatencies.size(); ++j) {
        std::string baseName = "imbalance_rp_" + std::to_string(pathIdx) + "_" +
                               std::to_string(i) + "_" + std::to_string(j);

        model->addConstr(pathLatencies[i] - pathLatencies[j] <=
                             BIG_M * patternImbalanced,
                         baseName + "_a");
        model->addConstr(pathLatencies[j] - pathLatencies[i] <=
                             BIG_M * patternImbalanced,
                         baseName + "_b");
      }

      // TODO(ziad): Uncomment for better II on circuits with high-latency
      // components. This +1 margin is NOT in the paper but helps achieve 2000
      // cycles with fir.c for some reason..... double maxBaseLat =
      //     *std::max_element(pathBaseLatencies.begin(),
      //     pathBaseLatencies.end());
      // double minBaseLat =
      //     *std::min_element(pathBaseLatencies.begin(),
      //     pathBaseLatencies.end());
      // bool hasLatencyImbalance = (maxBaseLat - minBaseLat) > 0.5;
      // if (hasLatencyImbalance) {
      //   model->addConstr(pathLatencies[i] >= maxBaseLat + 1,
      //                    "minLat_rp_" + std::to_string(pathIdx) + "_" +
      //                        std::to_string(i));
      // }
    }
  }
}

/// Helper to compute cycle latency expression:
/// Latency(cycle) = sum(D_u for units u) + sum(L_c for channels c)
/// (Paper: Section 4, Equation 1)
static LinExpr computeCycleLatency(const SimpleCycle &cycle,
                                   const SynchronizingCyclesFinderGraph &graph,
                                   const MILPVars &vars,
                                   const TimingDatabase &timingDB,
                                   double targetPeriod) {
  LinExpr latency;

  /// Add unit latencies D_u for each node in the cycle
  for (NodeIdType nodeId : cycle.nodes) {
    Operation *op = graph.nodes[nodeId].op;
    double unitLat = getUnitLatency(op, timingDB, targetPeriod);
    latency += unitLat;
  }

  /// Add extra latency variables L_c for edges in the cycle
  for (size_t i = 0; i < cycle.nodes.size(); ++i) {
    NodeIdType src = cycle.nodes[i];
    NodeIdType dst = cycle.nodes[(i + 1) % cycle.nodes.size()];
    for (EdgeIdType edgeId : graph.adjList[src]) {
      if (graph.edges[edgeId].dstId == dst) {
        Value channel = graph.edges[edgeId].channel;
        if (vars.channelVars.count(channel)) {
          latency += vars.channelVars.lookup(channel).extraLatency;
        }
        break;
      }
    }
  }

  return latency;
}

void LatencyBalancingMILP::addSyncCycleConstraints() {
  for (size_t pairIdx = 0; pairIdx < syncCyclePairs.size(); ++pairIdx) {
    const SynchronizingCyclePair &pair = syncCyclePairs[pairIdx];
    CPVar &patternImbalanced = vars.syncCycleVars[pairIdx].imbalanced;

    bool hasVarLatency =
        checkForVariableLatency(pair.cycleOne.nodes, syncGraph) ||
        checkForVariableLatency(pair.cycleTwo.nodes, syncGraph);

    /// If variable latency exists, force pattern to be marked imbalanced
    /// (Paper: Section 4, Equation 3)
    if (hasVarLatency) {
      model->addConstr(patternImbalanced == 1,
                       "varLatency_sc_" + std::to_string(pairIdx));
      continue;
    }

    /// Per (Paper: Section 4, Definition 3, Equation 2):
    /// For synchronizing cycles, we balance the CYCLE latencies themselves.
    /// The two cycles must have equal latency to be balanced.
    LinExpr latencyCycleOne = computeCycleLatency(pair.cycleOne, syncGraph,
                                                  vars, timingDB, targetPeriod);
    LinExpr latencyCycleTwo = computeCycleLatency(pair.cycleTwo, syncGraph,
                                                  vars, timingDB, targetPeriod);

    std::string baseName = "imbalance_sc_" + std::to_string(pairIdx);

    /// (Paper: Section 4, Equation 2)
    model->addConstr(latencyCycleOne - latencyCycleTwo <=
                         BIG_M * patternImbalanced,
                     baseName + "_a");
    model->addConstr(latencyCycleTwo - latencyCycleOne <=
                         BIG_M * patternImbalanced,
                     baseName + "_b");
  }
}

/// For each channel, link its stall status to the patterns' imbalance status.
/// (Paper: Section 4, Equation 4)
void LatencyBalancingMILP::addStallPropagationConstraints() {
  DenseMap<Value, SmallVector<CPVar *>> channelToPatterns;

  /// From reconvergent paths:
  for (size_t i = 0; i < reconvergentPaths.size(); ++i) {
    const ReconvergentPathWithGraph &pathWithGraph = reconvergentPaths[i];
    const ReconvergentPath &path = pathWithGraph.path;
    const ReconvergentPathFinderGraph *graph = pathWithGraph.graph;
    for (NodeIdType nodeId : path.nodeIds) {
      for (EdgeIdType edgeId : graph->adjList[nodeId]) {
        const auto &edge = graph->edges[edgeId];
        if (path.nodeIds.count(edge.dstId)) {
          channelToPatterns[edge.channel].push_back(
              &vars.reconvergentPathVars[i].imbalanced);
        }
      }
    }
  }

  /// From synchronizing cycle pairs:
  for (size_t i = 0; i < syncCyclePairs.size(); ++i) {
    const auto &pair = syncCyclePairs[i];
    for (const auto &edgeInfo : pair.edgesToJoins) {
      for (EdgeIdType edgeId : edgeInfo.edgesFromCycleOne) {
        channelToPatterns[syncGraph.edges[edgeId].channel].push_back(
            &vars.syncCycleVars[i].imbalanced);
      }
      for (EdgeIdType edgeId : edgeInfo.edgesFromCycleTwo) {
        channelToPatterns[syncGraph.edges[edgeId].channel].push_back(
            &vars.syncCycleVars[i].imbalanced);
      }
    }
  }

  /// For each channel, add: stalled_c >= s_p for all patterns p containing c
  /// (Paper: Section 4, Equation 4)
  size_t channelIdx = 0;
  for (auto &[channel, patterns] : channelToPatterns) {
    if (!vars.channelVars.count(channel)) {
      channelIdx++;
      continue;
    }

    CPVar &stalled = vars.channelVars[channel].stalled;
    for (size_t i = 0; i < patterns.size(); ++i) {
      std::string cstrName =
          "stallProp_" + std::to_string(channelIdx) + "_" + std::to_string(i);
      model->addConstr(stalled >= *patterns[i], cstrName);
    }
    channelIdx++;
  }
}

/// Compute the base latency of a cycle
/// (Paper: Section 4, Equation 1)
static double
computeCycleBaseLatency(const SimpleCycle &cycle,
                        const SynchronizingCyclesFinderGraph &graph,
                        const TimingDatabase &timingDB, double targetPeriod) {
  double latency = 0.0;
  for (NodeIdType nodeId : cycle.nodes) {
    Operation *op = graph.nodes[nodeId].op;
    latency += getUnitLatency(op, timingDB, targetPeriod);
  }
  return latency;
}

void LatencyBalancingMILP::addCycleTimeConstraints() {
  /// Per (Paper: Section 7):
  /// "First solve the formulation assuming II = 1, then 2, until feasible"
  ///
  /// Per (Paper: Section 4, Equation 5): Latency(l) <= II_CFC
  /// Per (Paper: Section 4, Equation 15): Latency(l) >= II_CFC (to set loop II
  /// exactly) Combined: Latency(l) == II_CFC for all cycles
  ///
  /// We start with II = 1. If any cycle has base latency > II, the LP is
  /// infeasible, and we need to increase II. We compute the minimum feasible
  /// II as max(1, max base cycle latency).

  if (cfdfcs.empty()) {
    llvm::errs() << "[LP1]   No CFDFCs, skipping cycle time constraints\n";
    return;
  }

  for (size_t cfdfcIdx = 0; cfdfcIdx < cfdfcs.size(); ++cfdfcIdx) {
    CFDFC *cfdfc = cfdfcs[cfdfcIdx];

    /// Build a graph for this CFDFC to find cycles
    SynchronizingCyclesFinderGraph cfdfcGraph;
    cfdfcGraph.buildFromCFDFC(funcInfo.funcOp, *cfdfc);
    std::vector<SimpleCycle> cycles = cfdfcGraph.findAllCycles();

    if (cycles.empty()) {
      LLVM_DEBUG(llvm::errs()
                 << "[LP1]   CFDFC " << cfdfcIdx << ": no cycles\n");
      continue;
    }

    double maxBaseLatency = 0.0;
    for (const auto &cycle : cycles) {
      double baseLatency =
          computeCycleBaseLatency(cycle, cfdfcGraph, timingDB, targetPeriod);
      maxBaseLatency = std::max(maxBaseLatency, baseLatency);
    }
    /// II_CFC = max(1, max base latency) - minimum feasible II
    double iiCFC = std::max(1.0, std::ceil(maxBaseLatency));

    /// Track the maximum II across all CFDFCs
    computedII = std::max(computedII, iiCFC);

    LLVM_DEBUG(llvm::errs()
               << "[LP1]   CFDFC " << cfdfcIdx << ": " << cycles.size()
               << " cycles, II_CFC = " << iiCFC
               << " (max base latency = " << maxBaseLatency << ")\n");

    /// Add constraints for each cycle: Latency(l) == II_CFC
    for (size_t cycleIdx = 0; cycleIdx < cycles.size(); ++cycleIdx) {
      const SimpleCycle &cycle = cycles[cycleIdx];

      /// Debug: Show channels on this cycle
      /// TODO(ziad): Remove this after making sure the LP is working correctly.
      LLVM_DEBUG({
        llvm::errs() << "[LP1]     Cycle " << cycleIdx << " channels: ";
        for (size_t i = 0; i < cycle.nodes.size(); ++i) {
          NodeIdType src = cycle.nodes[i];
          NodeIdType dst = cycle.nodes[(i + 1) % cycle.nodes.size()];
          for (EdgeIdType edgeId : cfdfcGraph.adjList[src]) {
            if (cfdfcGraph.edges[edgeId].dstId == dst) {
              Value channel = cfdfcGraph.edges[edgeId].channel;
              llvm::errs() << getUniqueName(*channel.getUses().begin()) << " ";
              break;
            }
          }
        }
        llvm::errs() << "\n";
      });

      /// Compute cycle latency expression: sum(D_u) + sum(L_c)
      LinExpr cycleLatency =
          computeCycleLatency(cycle, cfdfcGraph, vars, timingDB, targetPeriod);

      std::string baseName = "cycleTime_cfdfc" + std::to_string(cfdfcIdx) +
                             "_cycle" + std::to_string(cycleIdx);

      /// (Paper: Section 7, Equation 15) Latency(l) >= II_CFC (ensures all
      /// cycles have at least II_CFC)
      model->addConstr(cycleLatency >= iiCFC, baseName + "_min");
      /// (Paper: Section 4, Equation 5) Latency(l) <= II_CFC (ensures we don't
      /// exceed II_CFC)
      model->addConstr(cycleLatency <= iiCFC, baseName + "_max");
    }
  }
}

/// Objective per (Paper: Section 4, Equation 7):
/// Minimize STALL_WEIGHT*sum(S_c) + LATENCY_WEIGHT*sum(B_c*R_c + L_c)
/// Where STALL_WEIGHT >> LATENCY_WEIGHT to prioritize removing stalls
/// NOTE: STALL_WEIGHT and LATENCY_WEIGHT are defined as alpha and beta in the
/// paper.
/// - S_c: whether channel c is stalled
/// - B_c: bitwidth of channel c
/// - R_c: binary indicating if channel has latency (L_c > 0)
/// - L_c: extra latency on channel c
void LatencyBalancingMILP::setLatencyBalancingObjective() {

  LinExpr objective;

  for (auto &[channel, chVars] : vars.channelVars) {
    /// Primary objective (alpha term): minimize number of stalled channels
    objective += STALL_WEIGHT * chVars.stalled;

    /// Secondary objective (beta term): minimize latency cost
    /// B_c*R_c: fixed cost per channel cut, weighted by bitwidth
    unsigned bitwidth = getChannelBitwidth(channel);
    objective += LATENCY_WEIGHT * (bitwidth * chVars.bufPresent);
    /// L_c: linear cost of extra latency
    objective += LATENCY_WEIGHT * chVars.extraLatency;
  }

  /// Minimize by maximizing negative
  /// There is no .setMinimizeObjective() method, so we maximize the negative of
  /// the objective.
  /// TODO(ziad): Clarify with Jiahui if this is the correct approach.
  model->setMaximizeObjective(-objective);
}

// This method is required by the base class but not used since we feed the
// results to the occupancy balancing LP anyways.
void LatencyBalancingMILP::extractResult(BufferPlacement &placement) {}

/// Extract the latency results from the LP.
LatencyBalancingResult LatencyBalancingMILP::extractLatencyResults() {
  LatencyBalancingResult result;

  for (auto &[channel, chVars] : vars.channelVars) {
    unsigned extraLatency =
        static_cast<unsigned>(model->getValue(chVars.extraLatency) + 0.5);
    result.channelExtraLatency[channel] = extraLatency;

    LLVM_DEBUG(llvm::errs()
               << "Channel " << getUniqueName(*channel.getUses().begin())
               << ": extra_latency=" << extraLatency
               << ", stalled=" << model->getValue(chVars.stalled) << "\n");
  }

  result.targetII = computedII;
  LLVM_DEBUG(llvm::errs() << "[LP1] Computed target II = " << computedII
                          << "\n");
  return result;
}

/// OccupancyBalancingLP Implementation  ///

OccupancyBalancingLP::OccupancyBalancingLP(
    CPSolver::SolverKind solverKind, int timeout, FuncInfo &funcInfo,
    const TimingDatabase &timingDB, double targetPeriod,
    const LatencyBalancingResult &latencyResult,
    ArrayRef<ReconvergentPathWithGraph> reconvergentPaths,
    ArrayRef<CFDFC *> cfdfcs)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB,
                          targetPeriod),
      latencyResult(latencyResult), reconvergentPaths(reconvergentPaths),
      cfdfcs(cfdfcs) {
  setup();
}

void OccupancyBalancingLP::setup() {
  LLVM_DEBUG(llvm::errs() << "[LP2] Setting up Occupancy Balancing LP...\n");

  if (unsatisfiable)
    return;

  if (cfdfcs.empty()) {
    LLVM_DEBUG(llvm::errs() << "[LP2] WARNING: No CFDFCs provided\n");
    unsatisfiable = true;
    return;
  }

  /// Collect ALL channels from CFDFCs
  DenseSet<Value> allChannels;
  for (CFDFC *cfdfc : cfdfcs) {
    for (Value channel : cfdfc->channels) {
      allChannels.insert(channel);
    }
  }

  LLVM_DEBUG(llvm::errs() << "[LP2]   Found " << allChannels.size()
                          << " CFDFC channels\n");

  if (allChannels.empty()) {
    LLVM_DEBUG(llvm::errs() << "[LP2] WARNING: No channels found\n");
    unsatisfiable = true;
    return;
  }

  /// Target II = 1 for maximum throughput
  double targetII = latencyResult.targetII;
  if (targetII <= 0.0) {
    targetII = 1.0;
  }
  LLVM_DEBUG(llvm::errs() << "[LP2]   Target II = " << targetII << "\n");

  /// Create variables for each channel
  /// N_c: Maximal token occupancy on channel c.
  /// (Paper: Section 5, Table 2)
  for (Value channel : allChannels) {
    std::string name = getUniqueName(*channel.getUses().begin());
    CPVar var = model->addVar("n_" + name, CPVar::REAL, 0.0, MAX_OCCUPANCY);
    channelOccupancy[channel] = var;
  }
  LLVM_DEBUG(llvm::errs() << "[LP2]   Created " << channelOccupancy.size()
                          << " occupancy variables\n");

  /// (Paper: Section 5, Equation 8): N_c >= L_c / II
  size_t constraintCount = 0;
  for (Value channel : allChannels) {
    /// Get L_c from the latency balancing's results.
    unsigned latency = 0;
    if (latencyResult.channelExtraLatency.count(channel)) {
      latency = latencyResult.channelExtraLatency.lookup(channel);
    }
    double minOccupancy = static_cast<double>(latency) / targetII;

    model->addConstr(channelOccupancy[channel] >= minOccupancy,
                     "n_c>=(L_c/II)" +
                         getUniqueName(*channel.getUses().begin()));
    constraintCount++;
  }
  LLVM_DEBUG(llvm::errs() << "[LP2]   Added " << constraintCount
                          << " N_c >= L_c/II constraints\n");

  /// Path consistency constraints (Paper: Section 5, Equation 11)
  /// NOTE: The per-channel constraint N_c >= L_c / II from above
  /// should be sufficient when LP1 correctly balances latency. The path sum
  /// constraint can cause occupancy to concentrate on channels where latency
  /// wasn't added (e.g., on mux outputs instead of fork outputs), leading to
  /// transparent FIFOs instead of registers.
  ///
  /// For now, we rely on N_c >= L_c / II to distribute occupancy
  /// proportionally to where latency was added by LP1. This ensures DV buffers
  /// are placed where needed.
  ///
  /// TODO(ziad): Revisit path occupancy constraints. Equation 11
  /// says paths should have equal occupancy, but implementing this as a sum
  /// constraint allows the solver to concentrate occupancy arbitrarily.
  /// I have no idea why this is happening, but adding it makes everything
  /// worse.
  LLVM_DEBUG(
      llvm::errs()
      << "[LP2]   Skipping path sum constraints (relying on N_c >= L_c/II)\n");

  /// Add cycle capacity constraints
  /// (Paper: Section 5, Equation 12): Occupancy(cycle) <= B
  /// For sequential programs, B=1 means at most 1 token per cycle.
  /// We enforce N_c >= 1 for backedges to ensure each cycle has at least 1
  /// token, which combined with the minimization objective effectively limits
  /// to exactly 1 token for sequential execution.
  size_t cycleConstraints = 0;
  for (size_t i = 0; i < cfdfcs.size(); ++i) {
    CFDFC *cfdfc = cfdfcs[i];
    for (Value channel : cfdfc->backedges) {
      if (channelOccupancy.count(channel)) {
        model->addConstr(channelOccupancy[channel] >= 1.0,
                         "backedge_" + std::to_string(i));
        cycleConstraints++;
      }
    }
  }
  LLVM_DEBUG(llvm::errs() << "[LP2]   Added " << cycleConstraints
                          << " cycle capacity constraints\n");

  /// Set objective to minimize total buffer area
  /// (Paper: Section 5, Equation 14): Minimize sum(B_c*N_c)
  LinExpr objective;
  for (Value channel : allChannels) {
    unsigned bitwidth = handshake::getHandshakeTypeBitWidth(channel.getType());
    objective += bitwidth * channelOccupancy[channel];
  }
  /// Again, as above, we minimize by maximizing the negative.
  model->setMaximizeObjective(-objective);

  markReadyToOptimize();
  LLVM_DEBUG(llvm::errs() << "[LP2] Setup complete.\n");
}

void OccupancyBalancingLP::extractResult(BufferPlacement &placement) {
  LLVM_DEBUG(llvm::errs() << "[LP2] Extracting results...\n");

  for (auto &[channel, var] : channelOccupancy) {
    double occupancy = model->getValue(var);
    unsigned numSlots = static_cast<unsigned>(std::ceil(occupancy));

    /// Get L_c (latency) from the latency balancing's results.
    unsigned latencyCycles = 0;
    if (latencyResult.channelExtraLatency.count(channel)) {
      latencyCycles = latencyResult.channelExtraLatency.lookup(channel);
    }

    /// Ensure at least 1 slot if there's latency
    if (latencyCycles > 0 && numSlots == 0)
      numSlots = 1;

    /// Store occupancy in CFDFCs
    for (CFDFC *cfdfc : cfdfcs) {
      if (cfdfc->channels.contains(channel)) {
        cfdfc->channelOccupancy[channel] = occupancy;
      }
    }

    if (numSlots == 0 && latencyCycles == 0) {
      continue;
    }

    PlacementResult result;

    /// Buffer configuration (Paper: Section 6)
    /// L_c = latencyCycles (extra latency to add)
    /// N_c = numSlots (occupancy/capacity needed)
    if (latencyCycles == 0 && numSlots > 0) {
      /// Case 1: L=0, N>0 - No latency, just storage
      result.numFifoNone = numSlots;
    } else if (numSlots > latencyCycles) {
      /// Case 3: N > L - Need L pipeline stages + (N-L) transparent FIFO slots
      result.numOneSlotDV = latencyCycles;
      result.numFifoNone = numSlots - latencyCycles;
    } else {
      /// Case 2: L >= N - Need L cycles of latency
      /// Paper suggests ⌈L/N⌉ latency per slot, but our infrastructure
      /// uses 1-cycle DV buffers. We use L DV buffers for correctness.
      /// This provides L cycles latency and L capacity (>= N, so sufficient).
      result.numOneSlotDV = latencyCycles;
    }

    /// For Mux/Merge/ControlMerge on cycles, add break_r for deadlock
    /// prevention.
    Operation *srcOp = channel.getDefiningOp();
    if (isa_and_nonnull<handshake::MuxOp, handshake::MergeOp,
                        handshake::ControlMergeOp>(srcOp) &&
        srcOp->getNumOperands() > 1 && isChannelOnCycle(channel)) {
      result.numOneSlotR = 1;
      if (result.numOneSlotDV == 0)
        result.numOneSlotDV = 1;
    }

    if (result.numFifoNone > 0 || result.numOneSlotDV > 0 ||
        result.numOneSlotR > 0) {
      placement[channel] = result;
      LLVM_DEBUG(llvm::errs()
                 << "  " << getUniqueName(*channel.getUses().begin())
                 << ": L=" << latencyCycles << ", N=" << numSlots
                 << ", occ=" << occupancy << " -> DV=" << result.numOneSlotDV
                 << ", FIFO=" << result.numFifoNone
                 << ", R=" << result.numOneSlotR << "\n");
    }
  }
}

/// FPGA24Buffers Main Entry Point ///

FPGA24Buffers::FPGA24Buffers(CPSolver::SolverKind solverKind, int timeout,
                             FuncInfo &funcInfo, const TimingDatabase &timingDB,
                             double targetPeriod)
    : solverKind(solverKind), timeout(timeout), funcInfo(funcInfo),
      targetPeriod(targetPeriod), timingDB(timingDB) {}

LogicalResult FPGA24Buffers::solve(BufferPlacement &placement) {
  LLVM_DEBUG(llvm::errs() << "=== FPGA24 Buffer Placement ===\n");

  SmallVector<CFDFC *> cfdfcPtrs;
  for (auto &[cfdfc, _] : funcInfo.cfdfcs)
    cfdfcPtrs.push_back(cfdfc);

  LLVM_DEBUG(llvm::errs() << "Found " << cfdfcPtrs.size() << " CFDFCs\n");

  std::list<ReconvergentPathFinderGraph> reconvergentGraphs;
  std::vector<ReconvergentPathWithGraph> allReconvergentPaths;
  std::vector<SynchronizingCyclePair> allSyncCyclePairs;
  SynchronizingCyclesFinderGraph syncGraph;

  if (!cfdfcPtrs.empty()) {
    syncGraph.buildFromCFDFC(funcInfo.funcOp, *cfdfcPtrs[0]);
    allSyncCyclePairs = syncGraph.findSynchronizingCyclePairs();
    LLVM_DEBUG(llvm::errs() << "Found " << allSyncCyclePairs.size()
                            << " synchronizing cycle pairs\n");
  }

  const auto &archTransitions = funcInfo.archs;

  if (!archTransitions.empty()) {
    constexpr size_t sequenceLength = 2;
    auto sequences =
        enumerateTransitionSequences(archTransitions, sequenceLength);

    LLVM_DEBUG(llvm::errs() << "Enumerated " << sequences.size()
                            << " transition sequences\n");

    std::set<std::pair<Operation *, Operation *>> seenForkJoinPairs;

    size_t totalSequences = sequences.size();
    size_t duplicatesSkipped = 0;
    for (size_t seqIdx = 0; seqIdx < totalSequences; ++seqIdx) {
      if (seqIdx % 50 == 0 || seqIdx == totalSequences - 1) {
        LLVM_DEBUG(llvm::errs()
                   << "  Processing sequence " << seqIdx + 1 << "/"
                   << totalSequences << " (found "
                   << allReconvergentPaths.size() << " unique paths, "
                   << duplicatesSkipped << " duplicates skipped)\n");
      }

      const auto &sequence = sequences[seqIdx];
      ReconvergentPathFinderGraph graph;
      graph.buildGraphFromSequence(funcInfo.funcOp, sequence);
      auto paths = graph.findReconvergentPaths();

      if (!paths.empty()) {
        /// Filter out duplicate fork/join pairs we've already seen
        std::vector<ReconvergentPath> uniquePaths;
        for (auto &path : paths) {
          Operation *forkOp = graph.nodes[path.forkNodeId].op;
          Operation *joinOp = graph.nodes[path.joinNodeId].op;
          auto key = std::make_pair(forkOp, joinOp);

          if (seenForkJoinPairs.count(key)) {
            duplicatesSkipped++;
            continue;
          }
          seenForkJoinPairs.insert(key);
          uniquePaths.push_back(std::move(path));
        }

        if (!uniquePaths.empty()) {
          /// Add graph to list first
          reconvergentGraphs.push_back(std::move(graph));
          const ReconvergentPathFinderGraph *graphPtr =
              &reconvergentGraphs.back();

          /// Then add paths with pointer to their graph
          for (auto &path : uniquePaths) {
            allReconvergentPaths.emplace_back(std::move(path), graphPtr);
          }
        }
      }
    }

    LLVM_DEBUG(llvm::errs() << "Found " << allReconvergentPaths.size()
                            << " unique reconvergent paths across "
                            << reconvergentGraphs.size() << " graphs ("
                            << duplicatesSkipped << " duplicates skipped)\n");
  }

  /// Solve Latency Balancing LP1
  LLVM_DEBUG(llvm::errs() << "=== Setting up LP1 (Latency Balancing) ===\n");

  LatencyBalancingMILP latencyBalancingLP(
      solverKind, timeout, funcInfo, timingDB, targetPeriod,
      allReconvergentPaths, allSyncCyclePairs, syncGraph, cfdfcPtrs);

  LLVM_DEBUG(llvm::errs() << "=== Optimizing LP1 ===\n");
  if (failed(latencyBalancingLP.optimize())) {
    LLVM_DEBUG(llvm::errs() << "LP1 optimization failed\n");
    return failure();
  }
  LLVM_DEBUG(llvm::errs() << "LP1 optimization complete.\n");

  LatencyBalancingResult latencyResult =
      latencyBalancingLP.extractLatencyResults();
  LLVM_DEBUG(llvm::errs() << "LP1 computed extra latencies for "
                          << latencyResult.channelExtraLatency.size()
                          << " channels\n");

  /// Debug: Verify cycle latencies after LP1
  /// TODO(ziad): Remove this after making sure the LP is working correctly.
  LLVM_DEBUG({
    llvm::errs() << "=== Verifying CFDFC Cycle Latencies After LP1 ===\n";
    for (size_t cfdfcIdx = 0; cfdfcIdx < cfdfcPtrs.size(); ++cfdfcIdx) {
      CFDFC *cfdfc = cfdfcPtrs[cfdfcIdx];
      SynchronizingCyclesFinderGraph cfdfcGraph;
      cfdfcGraph.buildFromCFDFC(funcInfo.funcOp, *cfdfc);
      std::vector<SimpleCycle> cycles = cfdfcGraph.findAllCycles();

      for (size_t cycleIdx = 0; cycleIdx < cycles.size(); ++cycleIdx) {
        const SimpleCycle &cycle = cycles[cycleIdx];
        unsigned totalLatency = 0;
        llvm::errs() << "  Cycle " << cycleIdx << ": ";
        for (size_t i = 0; i < cycle.nodes.size(); ++i) {
          NodeIdType src = cycle.nodes[i];
          NodeIdType dst = cycle.nodes[(i + 1) % cycle.nodes.size()];
          for (EdgeIdType edgeId : cfdfcGraph.adjList[src]) {
            if (cfdfcGraph.edges[edgeId].dstId == dst) {
              Value channel = cfdfcGraph.edges[edgeId].channel;
              unsigned extraLat = 0;
              if (latencyResult.channelExtraLatency.count(channel)) {
                extraLat = latencyResult.channelExtraLatency.lookup(channel);
              }
              if (extraLat > 0) {
                llvm::errs() << getUniqueName(*channel.getUses().begin())
                             << "(L=" << extraLat << ") ";
              }
              totalLatency += extraLat;
              break;
            }
          }
        }
        llvm::errs() << "-> Total cycle latency = " << totalLatency << "\n";
        if (totalLatency > 1) {
          llvm::errs() << "  WARNING: Cycle " << cycleIdx
                       << " has latency > 1, will cause II > 1!\n";
        }
      }
    }
  });

  /// Solve Occupancy Balancing LP2
  LLVM_DEBUG(llvm::errs() << "=== Setting up LP2 (Occupancy Balancing) ===\n");
  /// Key equations:
  /// - (Paper: Section 5, Equation 8): N_c >= L_c / II
  /// - (Paper: Section 5, Equation 10): Occupancy(p) = sum(N_u) + sum(N_c)
  /// (path occupancy includes units)
  /// - (Paper: Section 5, Equation 11): Occupancy(p1) = Occupancy(p2) for
  /// reconvergent paths
  /// - (Paper: Section 5, Equation 12): Occupancy(cycle) <= B where B=1 for
  /// sequential programs
  LLVM_DEBUG(llvm::errs() << "=== Occupancy Balancing (LP2) ===\n");

  OccupancyBalancingLP occupancyBalancingLP(
      solverKind, timeout, funcInfo, timingDB, targetPeriod, latencyResult,
      allReconvergentPaths, cfdfcPtrs);

  if (failed(occupancyBalancingLP.optimize())) {
    LLVM_DEBUG(llvm::errs() << "LP2 optimization failed\n");
    return failure();
  }

  LLVM_DEBUG(llvm::errs() << "LP2 optimization complete.\n");
  occupancyBalancingLP.extractResult(placement);

  /// Post-process: Add DV+R for Mux/Merge/ControlMerge for deadlock prevention.
  for (CFDFC *cfdfc : cfdfcPtrs) {
    for (Value channel : cfdfc->channels) {
      Operation *srcOp = channel.getDefiningOp();
      bool isMergeLike = isa_and_nonnull<handshake::MuxOp, handshake::MergeOp,
                                         handshake::ControlMergeOp>(srcOp);

      if (isMergeLike) {
        PlacementResult &result = placement[channel];
        if (result.numOneSlotDV == 0)
          result.numOneSlotDV = 1;
        result.numOneSlotR = 1;
        llvm::errs() << "  Adding DV+R for merge-like: "
                     << getUniqueName(*channel.getUses().begin()) << "\n";
      }
    }
  }

  /// Debug: Log final placed buffers
  /// TODO(ziad): Remove this after making sure the LP is working correctly.
  LLVM_DEBUG(llvm::errs() << "Final buffer placement:\n");
  for (auto &[channel, result] : placement) {
    if (result.numOneSlotDV > 0 || result.numFifoNone > 0 ||
        result.numOneSlotR > 0) {
      LLVM_DEBUG(llvm::errs()
                 << "  " << getUniqueName(*channel.getUses().begin()) << ": DV="
                 << result.numOneSlotDV << ", FIFO=" << result.numFifoNone
                 << ", R=" << result.numOneSlotR << "\n");
    }
  }

  /// Populate channelOccupancy for all CFDFC channels (required by
  /// instantiateBuffers) Channels we placed buffers on get their computed
  /// occupancy, others get 0.0
  for (CFDFC *cfdfc : cfdfcPtrs) {
    for (Value channel : cfdfc->channels) {
      if (cfdfc->channelOccupancy.count(channel) == 0) {
        /// Default occupancy for channels not involved in our analysis
        cfdfc->channelOccupancy[channel] = 0.0;
      }
    }
  }

  LLVM_DEBUG(llvm::errs() << "=== FPGA24 Buffer Placement Complete ===\n");
  LLVM_DEBUG(llvm::errs() << "Placed buffers on " << placement.size()
                          << " channels\n");

  return success();
}
