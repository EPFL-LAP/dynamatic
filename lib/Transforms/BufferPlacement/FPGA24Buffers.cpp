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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <list>
#include <set>
#include <string>

using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpga24;

#define DEBUG_TYPE "fpga24-buffers"

/// LatencyBalancingMILP Implementation ///

LatencyBalancingMILP::LatencyBalancingMILP(
    CPSolver::SolverKind solverKind, int timeout, FuncInfo &funcInfo,
    const TimingDatabase &timingDB, double targetPeriod,
    ArrayRef<ReconvergentPathWithGraph> reconvergentPaths,
    ArrayRef<SynchronizingCyclePair> syncCyclePairs,
    const SynchronizingCyclesFinderGraph &syncGraph, ArrayRef<CFDFC *> cfdfcs)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB,
                          targetPeriod),
      reconvergentPaths(reconvergentPaths.begin(), reconvergentPaths.end()),
      syncCyclePairs(syncCyclePairs.begin(), syncCyclePairs.end()),
      syncGraph(syncGraph), cfdfcs(cfdfcs.begin(), cfdfcs.end()) {
  setup();
}

void LatencyBalancingMILP::setup() {
  if (unsatisfiable)
    return;

  addLatencyBalancingVars(reconvergentPaths, syncCyclePairs);
  addChannelPropertyLatencyConstraints();
  addReconvergentPathConstraints(reconvergentPaths);
  addSyncCycleConstraints(syncCyclePairs, syncGraph);
  addStallPropagationConstraints(reconvergentPaths, syncCyclePairs, syncGraph);
  addCycleTimeConstraints(cfdfcs, computedII, computedCFDFCIIs);
  setLatencyBalancingObjective();
  markReadyToOptimize();
}

void LatencyBalancingMILP::addChannelPropertyLatencyConstraints() {
  for (auto &[channel, chVars] : vars.channelVars) {
    handshake::ChannelBufProps &props = channelProps[channel];
    std::string name = getUniqueName(*channel.getUses().begin());

    /// As in `FPGA20Buffers::addCustomChannelConstraints`, `minOpaque` only
    /// forces the binary "data/valid is broken" decision. The total slot count
    /// is handled in the occupancy LP.
    if (props.minOpaque > 0) {
      model->addConstr(chVars.bufPresent == 1, "fpga24_forceOpaque_R_" + name);
    }

    if (props.maxOpaque.has_value() && *props.maxOpaque == 0) {
      model->addConstr(chVars.dataLatency == 0,
                       "fpga24_forceTransparent_L_" + name);
      model->addConstr(chVars.bufPresent == 0,
                       "fpga24_forceTransparent_R_" + name);
    }
  }
}

/// The latency variable L_c is the number of extra latencies to be added to a
/// channel. It will be used in the input of the occupancy balancing LP. Defined
/// in (Paper: Section 4, Table 1).
// This method is required by the base class but not used since we feed the
// results to the occupancy balancing LP anyways.
void LatencyBalancingMILP::extractResult(BufferPlacement &placement) {}

/// Extract the latency results from the LP.
LatencyBalancingResult LatencyBalancingMILP::extractLatencyResults() {
  LatencyBalancingResult result;

  for (auto &[channel, chVars] : vars.channelVars) {
    unsigned dataLatency =
        static_cast<unsigned>(model->getValue(chVars.dataLatency) + 0.5);
    result.channelExtraLatency[channel] = dataLatency;
  }

  result.targetII = computedII;
  result.cfdfcTargetIIs = computedCFDFCIIs;
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
      latencyResult(latencyResult),
      reconvergentPaths(reconvergentPaths.begin(), reconvergentPaths.end()),
      cfdfcs(cfdfcs.begin(), cfdfcs.end()) {
  setup();
}

void OccupancyBalancingLP::setup() {
  if (unsatisfiable)
    return;

  if (cfdfcs.empty()) {
    unsatisfiable = true;
    return;
  }

  SmallVector<Value> allChannels;
  for (auto &[channel, _] : channelProps) {
    if (isa<MemRefType>(channel.getType()))
      continue;
    allChannels.push_back(channel);
  }
  if (allChannels.empty()) {
    unsatisfiable = true;
    return;
  }

  /// Target II = 1 for maximum throughput
  double targetII = latencyResult.targetII;
  if (targetII <= 0.0) {
    targetII = 1.0;
  }
  /// Create variables for each channel
  /// N_c: Maximal token occupancy on channel c.
  /// (Paper: Section 5, Table 2)
  this->addOccupancyVars(allChannels, channelOccupancy, MAX_OCCUPANCY);

  /// (Paper: Section 5, Equation 8): N_c >= L_c / II
  /// We enforce this for the global II, but also for each CFDFC's specific II
  /// to ensure sufficient buffering in faster loops.
  /// (Paper: Section 5, Equation 15): Making the required occupancy the
  /// maximum of all CFDFCs' II.

  DenseMap<Value, double> requiredOccupancy;

  // Initialize with global II constraint
  for (Value channel : allChannels) {
    unsigned latency = latencyResult.channelExtraLatency.lookup(channel);
    requiredOccupancy[channel] = static_cast<double>(latency) / targetII;
  }

  // Update by taking the maximum of the per-CFDFC constraints
  for (CFDFC *cfdfc : cfdfcs) {
    double cfdfcII = latencyResult.cfdfcTargetIIs.lookup(cfdfc);
    if (cfdfcII < 1.0)
      continue;

    for (Value channel : cfdfc->channels) {
      if (requiredOccupancy.count(channel)) {
        unsigned latency = latencyResult.channelExtraLatency.lookup(channel);
        double specificOcc = static_cast<double>(latency) / cfdfcII;
        if (specificOcc > requiredOccupancy[channel]) {
          requiredOccupancy[channel] = specificOcc;
        }
      }
    }
  }

  addMinOccupancyConstraints(requiredOccupancy, channelOccupancy);
  addBackedgeConstraints(cfdfcs, channelOccupancy);
  addChannelPropertyOccupancyConstraints(allChannels, channelOccupancy);

  this->setOccupancyBalancingObjective(allChannels, channelOccupancy);

  markReadyToOptimize();
}

void OccupancyBalancingLP::addChannelPropertyOccupancyConstraints(
    ArrayRef<Value> channels, DenseMap<Value, CPVar> &channelOccupancy) {
  for (Value channel : channels) {
    if (!channelOccupancy.count(channel))
      continue;
    handshake::ChannelBufProps &props = channelProps[channel];
    std::string name = getUniqueName(*channel.getUses().begin());
    CPVar &n = channelOccupancy[channel];

    bool hasOpaqueLatency =
        latencyResult.channelExtraLatency.lookup(channel) > 0;

    /// Same case split as `FPGA20Buffers::addCustomChannelConstraints`, with
    /// `hasOpaqueLatency` replacing FPGA20's binary data-buffer variable.
    if (props.minOpaque > 0) {
      if (props.minTrans > 0) {
        unsigned minTotal = props.minOpaque + props.minTrans;
        model->addConstr(n >= minTotal, "fpga24_minOpaqueAndTrans_N_" + name);
      } else {
        model->addConstr(n >= props.minOpaque, "fpga24_minOpaque_N_" + name);
      }
    } else if (props.minTrans > 0) {
      model->addConstr(n >= props.minTrans + (hasOpaqueLatency ? 1 : 0),
                       "fpga24_minTrans_N_" + name);
    } else if (props.minSlots > 0) {
      model->addConstr(n >= props.minSlots, "fpga24_minSlots_N_" + name);
    }

    if (props.maxOpaque.has_value() && props.maxTrans.has_value()) {
      unsigned maxSlots = *props.maxOpaque + *props.maxTrans;
      if (maxSlots == 0) {
        model->addConstr(n == 0, "fpga24_noSlots_N_" + name);
      } else {
        model->addConstr(n <= maxSlots, "fpga24_maxSlots_N_" + name);
      }
    }
  }
}

void OccupancyBalancingLP::extractResult(BufferPlacement &placement) {
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

    /// Buffer configuration with COUNTER_BUFFER:
    /// L_c = latencyCycles (extra latency to add)
    /// N_c = numSlots (occupancy/capacity needed)
    ///
    /// Counter Buffer Placement Logic:
    /// - place K counter buffers where K is bounded by both latency and
    ///   occupancy requirements;
    /// - distribute total latency across these K buffers so that
    ///   sum(dvLatency_i) = L_c;
    /// - add FIFO_BREAK_NONE slots when occupancy requires more pure storage.
    if (latencyCycles == 0 && numSlots > 0) {
      /// Case 1: L=0, N>0 - No latency, just storage
      result.numFifoNone = numSlots;
    } else if (latencyCycles > 0) {
      /// Case 2/3: L>0
      unsigned kCounter = std::max(1u, std::min(latencyCycles, numSlots));
      unsigned baseDelay = latencyCycles / kCounter;
      unsigned remainder = latencyCycles % kCounter;

      for (unsigned i = 0; i < kCounter; ++i) {
        unsigned delay = baseDelay + (i < remainder ? 1u : 0u);
        if (delay > 0)
          result.counterBufferLatencies.push_back(delay);
      }

      /// If occupancy requires more one-token buffers than those used to carry
      /// latency, add transparent slots for pure storage.
      if (numSlots > kCounter)
        result.numFifoNone = numSlots - kCounter;
    }

    /// For Mux/Merge/ControlMerge on cycles, add break_r for deadlock
    /// prevention.
    Operation *srcOp = channel.getDefiningOp();
    if (isa_and_nonnull<handshake::MuxOp, handshake::MergeOp,
                        handshake::ControlMergeOp>(srcOp) &&
        srcOp->getNumOperands() > 1 && isChannelOnCycle(channel)) {
      result.numOneSlotR = 1;
      if (result.numOneSlotDV == 0 && result.counterBufferLatencies.empty())
        result.counterBufferLatencies.push_back(1);
    }

    if (result.numFifoNone > 0 || result.numOneSlotDV > 0 ||
        result.numOneSlotR > 0 || !result.counterBufferLatencies.empty()) {
      placement[channel] = result;
    }
  }
}

/// FPGA24Buffers Main Entry Point ///

FPGA24Buffers::FPGA24Buffers(CPSolver::SolverKind solverKind, int timeout,
                             FuncInfo &funcInfo, const TimingDatabase &timingDB,
                             double targetPeriod)
    : solverKind(solverKind), timeout(timeout), funcInfo(funcInfo),
      targetPeriod(targetPeriod), timingDB(timingDB) {}

void FPGA24Buffers::findSynchronizationPatterns(
    ArrayRef<CFDFC *> cfdfcs,
    std::list<CFGTransitionSequenceSubgraph> &reconvergentGraphs,
    std::vector<ReconvergentPathWithGraph> &allReconvergentPaths,
    std::vector<SynchronizingCyclePair> &allSyncCyclePairs,
    SynchronizingCyclesFinderGraph &syncGraph) {

  if (!cfdfcs.empty()) {
    syncGraph.buildFromCFDFC(funcInfo.funcOp, *cfdfcs[0]);
    allSyncCyclePairs = syncGraph.findSynchronizingCyclePairs();
  }

  const auto &archTransitions = funcInfo.archs;
  if (archTransitions.empty())
    return;

  constexpr size_t sequenceLength = 4;
  auto sequences =
      enumerateTransitionSequences(archTransitions, sequenceLength);

  std::set<std::pair<Operation *, Operation *>> seenForkJoinPairs;

  size_t totalSequences = sequences.size();
  for (size_t seqIdx = 0; seqIdx < totalSequences; ++seqIdx) {
    const auto &sequence = sequences[seqIdx];
    CFGTransitionSequenceSubgraph graph;
    graph.buildGraphFromSequence(funcInfo.funcOp, sequence);
    auto paths = graph.findReconvergentPaths();
    CFGTransitionSequenceSubgraph::GraphPathsForDumping graphPaths = {&graph,
                                                                      paths};
    graph.dumpAllReconvergentPaths(
        graphPaths, "reconvergent_graph_" + std::to_string(seqIdx) + ".dot");

    if (paths.empty())
      continue;

    std::vector<ReconvergentPath> uniquePaths;
    for (auto &path : paths) {
      Operation *forkOp = graph.nodes[path.forkNodeId].op;
      Operation *joinOp = graph.nodes[path.joinNodeId].op;
      auto key = std::make_pair(forkOp, joinOp);

      if (seenForkJoinPairs.count(key)) {
        continue;
      }
      seenForkJoinPairs.insert(key);
      uniquePaths.push_back(std::move(path));
    }

    if (!uniquePaths.empty()) {
      reconvergentGraphs.push_back(std::move(graph));
      const CFGTransitionSequenceSubgraph *graphPtr =
          &reconvergentGraphs.back();

      for (auto &path : uniquePaths) {
        allReconvergentPaths.emplace_back(std::move(path), graphPtr);
      }
    }
  }
}

FailureOr<LatencyBalancingResult> FPGA24Buffers::solveLatencyBalancing(
    ArrayRef<CFDFC *> cfdfcs,
    ArrayRef<ReconvergentPathWithGraph> reconvergentPaths,
    ArrayRef<SynchronizingCyclePair> syncCyclePairs,
    const SynchronizingCyclesFinderGraph &syncGraph) {

  LatencyBalancingMILP latencyBalancingLP(
      solverKind, timeout, funcInfo, timingDB, targetPeriod, reconvergentPaths,
      syncCyclePairs, syncGraph, cfdfcs);

  if (failed(latencyBalancingLP.optimize())) {
    return failure();
  }

  LatencyBalancingResult result = latencyBalancingLP.extractLatencyResults();

  LLVM_DEBUG({
    llvm::errs() << "=== Verifying CFDFC Cycle Latencies After LP1 ===\n";
    for (auto [cfdfcIdx, cfdfc] : llvm::enumerate(cfdfcs)) {
      SynchronizingCyclesFinderGraph cfdfcGraph(funcInfo.funcOp, *cfdfc);
      std::vector<SimpleCycle> cycles = cfdfcGraph.findAllCycles();

      for (auto [cycleIdx, cycle] : llvm::enumerate(cycles)) {
        unsigned totalLatency = 0;
        llvm::errs() << "  Cycle " << cycleIdx << ": ";
        for (size_t i = 0; i < cycle.nodes.size(); ++i) {
          NodeIdType src = cycle.nodes[i];
          NodeIdType dst = cycle.nodes[(i + 1) % cycle.nodes.size()];
          for (EdgeIdType edgeId : cfdfcGraph.adjList[src]) {
            if (cfdfcGraph.edges[edgeId].dstId == dst) {
              Value channel = cfdfcGraph.edges[edgeId].channel;
              unsigned extraLat = 0;
              if (result.channelExtraLatency.count(channel)) {
                extraLat = result.channelExtraLatency.lookup(channel);
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

  return result;
}

LogicalResult FPGA24Buffers::solveOccupancyBalancing(
    BufferPlacement &placement, ArrayRef<CFDFC *> cfdfcs,
    ArrayRef<ReconvergentPathWithGraph> reconvergentPaths,
    const LatencyBalancingResult &latencyResult) {

  OccupancyBalancingLP occupancyBalancingLP(
      solverKind, timeout, funcInfo, timingDB, targetPeriod, latencyResult,
      reconvergentPaths, cfdfcs);

  if (failed(occupancyBalancingLP.optimize())) {
    return failure();
  }

  occupancyBalancingLP.extractResult(placement);
  return success();
}

void FPGA24Buffers::addPostProcessingBuffers(BufferPlacement &placement,
                                             ArrayRef<CFDFC *> cfdfcs) {
  /// Add R for Mux/Merge/ControlMerge outputs for deadlock prevention.
  for (CFDFC *cfdfc : cfdfcs) {
    for (Value channel : cfdfc->channels) {
      Operation *srcOp = channel.getDefiningOp();
      bool isMergeLike = isa_and_nonnull<handshake::MuxOp, handshake::MergeOp,
                                         handshake::ControlMergeOp>(srcOp);

      if (isMergeLike) {
        PlacementResult &result = placement[channel];

        result.numOneSlotR = 1;
        llvm::errs() << "  Adding R for merge-like: "
                     << getUniqueName(*channel.getUses().begin()) << "\n";
      }
    }
  }

  /// Buffer the paths to EndOp (<out0> or <end>) that represent the function
  /// end. (The ones not directly produced by memory controllers.)
  auto *terminator = funcInfo.funcOp.getBodyBlock()->getTerminator();
  if (auto endOp = dyn_cast<handshake::EndOp>(terminator)) {
    for (Value operand : endOp->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (!producer)
        continue;

      // Skip memory-completion paths; they do not represent function end.
      if (isa<handshake::MemoryOpInterface>(producer))
        continue;

      PlacementResult &result = placement[operand];
      if (result.numFifoNone == 0 && result.numOneSlotDV == 0 &&
          result.counterBufferLatencies.empty()) {
        result.numFifoNone = 1;
      }
    }
  }
}

LogicalResult FPGA24Buffers::solve(BufferPlacement &placement) {
  SmallVector<CFDFC *> cfdfcPtrs;
  for (auto &[cfdfc, _] : funcInfo.cfdfcs)
    cfdfcPtrs.push_back(cfdfc);

  std::list<CFGTransitionSequenceSubgraph> reconvergentGraphs;
  std::vector<ReconvergentPathWithGraph> allReconvergentPaths;
  std::vector<SynchronizingCyclePair> allSyncCyclePairs;
  SynchronizingCyclesFinderGraph syncGraph;
  findSynchronizationPatterns(cfdfcPtrs, reconvergentGraphs,
                              allReconvergentPaths, allSyncCyclePairs,
                              syncGraph);

  FailureOr<LatencyBalancingResult> latencyResult = solveLatencyBalancing(
      cfdfcPtrs, allReconvergentPaths, allSyncCyclePairs, syncGraph);
  if (failed(latencyResult))
    return failure();

  if (failed(solveOccupancyBalancing(placement, cfdfcPtrs, allReconvergentPaths,
                                     *latencyResult)))
    return failure();

  addPostProcessingBuffers(placement, cfdfcPtrs);

  LLVM_DEBUG({
    llvm::errs() << "Final buffer placement:\n";
    for (auto &[channel, result] : placement) {
      if (result.numOneSlotDV > 0 || result.numFifoNone > 0 ||
          result.numOneSlotR > 0) {
        llvm::errs() << "  " << getUniqueName(*channel.getUses().begin())
                     << ": DV=" << result.numOneSlotDV
                     << ", FIFO=" << result.numFifoNone
                     << ", R=" << result.numOneSlotR << "\n";
      }
    }
  });

  for (CFDFC *cfdfc : cfdfcPtrs) {
    for (Value channel : cfdfc->channels) {
      if (cfdfc->channelOccupancy.count(channel) == 0)
        cfdfc->channelOccupancy[channel] = 0.0;
    }
  }

  return success();
}
