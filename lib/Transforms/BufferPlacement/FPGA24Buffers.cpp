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
#include "dynamatic/Transforms/BufferPlacement/LatencyAndOccupancyBalancingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
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
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB, targetPeriod,
                          Algorithm::FPGA24),
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
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB, targetPeriod,
                          Algorithm::FPGA24),
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

  double targetII = latencyResult.targetII;
  if (targetII <= 0.0) {
    targetII = 1.0;
  }

  /// (Paper: Section 5, Table 2)
  /// N_c: Maximal token occupancy on channel c.
  this->addOccupancyVars(allChannels, cfdfcs, cfdfcChannelOccupancy,
                         cfdfcUnitOccupancy, maxChannelOccupancy,
                         MAX_OCCUPANCY);

  addMinOccupancyConstraints(cfdfcs, latencyResult.cfdfcTargetIIs,
                             latencyResult.channelExtraLatency,
                             cfdfcChannelOccupancy);
  addUnitOccupancyConstraints(cfdfcs, latencyResult.cfdfcTargetIIs,
                              cfdfcUnitOccupancy);
  addMaxOccupancyConstraints(cfdfcChannelOccupancy, maxChannelOccupancy);
  addBackedgeConstraints(cfdfcs, maxChannelOccupancy);
  addChannelPropertyOccupancyConstraints(maxChannelOccupancy);

  /// (Paper: Section 5, Equations 10-11): Path occupancy equality for
  /// reconvergent-path pairs inside each CFDFC.
  addPathOccupancyEqualityConstraints(
      reconvergentPaths, cfdfcs, cfdfcChannelOccupancy, cfdfcUnitOccupancy);

  addCycleOccupancyConstraints(cfdfcs, cfdfcChannelOccupancy,
                               cfdfcUnitOccupancy);

  /// (Paper: Section 5, Equation 14): Minimize sum(B_c * N_c).
  this->setOccupancyBalancingObjective(maxChannelOccupancy);

  markReadyToOptimize();
}

void OccupancyBalancingLP::addChannelPropertyOccupancyConstraints(
    llvm::MapVector<Value, CPVar> &channelOccupancy) {
  for (auto &[channel, n] : channelOccupancy) {
    handshake::ChannelBufProps &props = channelProps[channel];
    std::string name = getUniqueName(*channel.getUses().begin());

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
  /// Smallest CFC II; Case 2 slots must not be slower than this
  /// (Paper: Section 6).
  unsigned iiMin = 1;
  bool haveII = false;
  for (const auto &iiEntry : latencyResult.cfdfcTargetIIs) {
    double ii = iiEntry.second;
    if (ii < 1.0)
      continue;
    unsigned rounded = static_cast<unsigned>(ii + 0.5);
    if (!haveII || rounded < iiMin) {
      iiMin = std::max(1u, rounded);
      haveII = true;
    }
  }
  if (!haveII && latencyResult.targetII >= 1.0)
    iiMin = static_cast<unsigned>(latencyResult.targetII + 0.5);

  for (auto &[cfdfc, unitMap] : cfdfcUnitOccupancy) {
    for (auto &[unit, var] : unitMap)
      cfdfc->bufferOccupancy[unit] = model->getValue(var);
  }

  for (auto &[channel, var] : maxChannelOccupancy) {
    double occupancy = model->getValue(var);
    unsigned numSlots = static_cast<unsigned>(std::ceil(occupancy));

    unsigned latencyCycles = 0;
    if (latencyResult.channelExtraLatency.count(channel)) {
      latencyCycles = latencyResult.channelExtraLatency.lookup(channel);
    }

    /// Case 1 (Paper: Section 6): N=0, L>0 — one slot with all of L, no II cap.
    bool latencyOnly = latencyCycles > 0 && numSlots == 0;
    if (latencyOnly)
      numSlots = 1;

    for (CFDFC *cfdfc : cfdfcs) {
      if (!cfdfc->channels.contains(channel))
        continue;
      auto *cfcIt = cfdfcChannelOccupancy.find(cfdfc);
      if (cfcIt == cfdfcChannelOccupancy.end())
        continue;
      auto *chIt = cfcIt->second.find(channel);
      cfdfc->channelOccupancy[channel] =
          chIt != cfcIt->second.end() ? model->getValue(chIt->second) : 0.0;
    }

    if (numSlots == 0 && latencyCycles == 0) {
      continue;
    }

    PlacementResult result;

    /// Buffer configuration (Paper: Section 6):
    /// Case 1: L>0, N=0 → one L-cycle slot.
    /// Case 2: 0 < N <= L → N slots, split L, each slot delay <= II_min.
    /// Case 3: N > L → L slots of delay 1, plus N-L transparent slots.
    if (latencyCycles == 0 && numSlots > 0) {
      result.numFifoNone = numSlots;
    } else if (latencyCycles > 0) {
      unsigned kCounter = std::max(1u, std::min(latencyCycles, numSlots));
      if (!latencyOnly && numSlots <= latencyCycles) {
        unsigned minSlotsForII = (latencyCycles + iiMin - 1) / iiMin;
        kCounter = std::max(kCounter, minSlotsForII);
        kCounter = std::min(kCounter, latencyCycles);
      }

      unsigned baseDelay = latencyCycles / kCounter;
      unsigned remainder = latencyCycles % kCounter;

      for (unsigned i = 0; i < kCounter; ++i) {
        unsigned delay = baseDelay + (i < remainder ? 1u : 0u);
        if (delay > 0)
          result.counterBufferLatencies.push_back(delay);
      }

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
        LLVM_DEBUG({ llvm::errs() << "  Cycle " << cycleIdx << ": "; });
        auto findChannel = [&](NodeIdType src, NodeIdType dst) {
          for (EdgeIdType edgeId : cfdfcGraph.adjList[src]) {
            if (cfdfcGraph.edges[edgeId].dstId != dst)
              continue;
            return cfdfcGraph.edges[edgeId].channel;
          }
          llvm_unreachable("Edge not found");
        };

        for (size_t i = 0; i < cycle.nodes.size(); ++i) {
          NodeIdType src = cycle.nodes[i];
          NodeIdType dst = cycle.nodes[(i + 1) % cycle.nodes.size()];
          Value channel = findChannel(src, dst);

          unsigned extraLat = 0;
          if (result.channelExtraLatency.count(channel)) {
            extraLat = result.channelExtraLatency.lookup(channel);
          }
          if (extraLat > 0) {
            llvm::errs() << getUniqueName(*channel.getUses().begin())
                         << "(L=" << extraLat << ") ";
          }
          totalLatency += extraLat;
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

  /// It is hard to accurately model when memory controllers emit a "done"
  /// signal, which synchronizes with other function outputs. To prevent the
  /// backpressure to the function outputs from propagating into the internal
  /// logic,
  ///  we buffer the paths to EndOp (<out0> or <end>) that represent the
  ///  function end.
  /// (The ones not directly produced by memory controllers.)
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
