//===- FPGA24Buffers.h ------------------------------------------------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
//
// This algorithm is based on the paper: [Xu, Josipović, FPGA'24]
// (https://dl.acm.org/doi/10.1145/3626202.3637570)
// The paper implements a two-step approach to buffer placement:
// 1. Latency Balancing: This LP can be used to balance the latency of the
// circuit by calculating the extra latency to be added to each channel in order
// to equalize the latency of all paths in the circuit.
// 2. Occupancy Balancing: It takes the results of the latency balancing LP and
// uses them to calculate the number of slots to be used for each channel in
// order to balance the occupancy of the circuit.
//
// We run these LPs on two specific patterns: reconvergent paths and
// synchronizing cycles. Their definitions & implementations are provided in the
// `LatencyAndOccupancyBalancingSupport.{h,cpp}` file.
//
//===------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPGA24BUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPGA24BUFFERS_H

#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/LatencyAndOccupancyBalancingSupport.h"
#include <vector>

namespace dynamatic {
namespace buffer {
namespace fpga24 {

/// Constants ///

/// Big-M constant for imbalance constraints.
// (Paper: Section 4, Equation 2)
static constexpr double BIG_M = 1000.0;

/// Weight for stall penalty vs latency cost (>> LATENCY_WEIGHT to prioritize
/// stalls). See usage in (Paper: Section 4, Equation 7)
static constexpr double LATENCY_WEIGHT = 1.0;
static constexpr double STALL_WEIGHT = 1000.0;

/// Upper bound for occupancy
static constexpr double MAX_OCCUPANCY = 100.0;

/// Latency Balancing MILP ///

/// Holds the result of the first LP for usage in the LP.
struct LatencyBalancingResult {
  /// Map from channel to its computed extra latency.
  llvm::MapVector<Value, unsigned> channelExtraLatency;
  /// Target intiation interval.
  double targetII;
  /// Target intiation interval per CFDFC.
  llvm::MapVector<CFDFC *, double> cfdfcTargetIIs;
};

/// Helper struct that pairs a reconvergent path with its corresponding
/// transition graph. Since we're enumerating transition sequences, build graphs
/// from those and then in turn build reconvergent paths from those graphs, it's
/// better to pair them like this instead of playing around with indices.
struct ReconvergentPathWithGraph {
  ReconvergentPath path;
  const CFGTransitionSequenceSubgraph *graph;

  ReconvergentPathWithGraph(ReconvergentPath p,
                            const CFGTransitionSequenceSubgraph *g)
      : path(std::move(p)), graph(g) {}
};

class LatencyBalancingMILP : public BufferPlacementMILP {
public:
  LatencyBalancingMILP(CPSolver::SolverKind solverKind, int timeout,
                       FuncInfo &funcInfo, const TimingDatabase &timingDB,
                       double targetPeriod,
                       ArrayRef<ReconvergentPathWithGraph> reconvergentPaths,
                       ArrayRef<SynchronizingCyclePair> syncCyclePairs,
                       const SynchronizingCyclesFinderGraph &syncGraph,
                       ArrayRef<CFDFC *> cfdfcs);

  /// Extract latency results after solving.
  LatencyBalancingResult extractLatencyResults();

protected:
  /// Interpret solution - not used for buffer placement directly.
  void extractResult(BufferPlacement &placement) override;

private:
  std::vector<ReconvergentPathWithGraph> reconvergentPaths;
  std::vector<SynchronizingCyclePair> syncCyclePairs;

  /// Reference to synchronizing cycles graph.
  const SynchronizingCyclesFinderGraph &syncGraph;

  /// CFDFCs needed for cylce constraints.
  std::vector<CFDFC *> cfdfcs;

  /// Computed minimum feasible Initiation Interval across all CFDFCs (set by
  /// addCycleTimeConstraints).
  double computedII = 1.0;

  /// Computed minimum feasible Initiation Interval per CFDFC.
  llvm::MapVector<CFDFC *, double> computedCFDFCIIs;

  void addLatencyVariables();

  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

class OccupancyBalancingLP : public BufferPlacementMILP {
public:
  OccupancyBalancingLP(CPSolver::SolverKind solverKind, int timeout,
                       FuncInfo &funcInfo, const TimingDatabase &timingDB,
                       double targetPeriod,
                       const LatencyBalancingResult &latencyResult,
                       ArrayRef<ReconvergentPathWithGraph> reconvergentPaths,
                       ArrayRef<CFDFC *> cfdfcs);

  bool isUnsatisfiable() const { return unsatisfiable; }

  void extractResult(BufferPlacement &placement) override;

private:
  const LatencyBalancingResult &latencyResult;

  std::vector<ReconvergentPathWithGraph> reconvergentPaths;
  std::vector<CFDFC *> cfdfcs;

  DenseMap<Value, CPVar> channelOccupancy;

  void setup();
};

class FPGA24Buffers {
public:
  FPGA24Buffers(CPSolver::SolverKind solverKind, int timeout,
                FuncInfo &funcInfo, const TimingDatabase &timingDB,
                double targetPeriod);

  /// Run the complete algorithm and return placement.
  LogicalResult solve(BufferPlacement &placement);

private:
  CPSolver::SolverKind solverKind;
  int timeout;
  FuncInfo &funcInfo;
  double targetPeriod;
  const TimingDatabase &timingDB;
};

} // namespace fpga24
} // namespace buffer
} // namespace dynamatic

#endif /// DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPGA24BUFFERS_H