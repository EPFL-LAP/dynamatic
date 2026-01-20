//===- FPGA24Buffers.h ------------------------------------------------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
//
// TODO
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

namespace dynamatic {
namespace buffer {
namespace fpga24 {

/// Latency Balancing MILP ///

/// Holds the result of the first LP for usage in the LP.
struct LatencyBalancingResult {
  /// Map from channel to its computed extra latency.
  DenseMap<Value, unsigned> channelExtraLatency;
  /// Target intiation interval.
  double targetII;
};

/// Helper struct that pairs a reconvergent path with its corresponding
/// transition graph. Since we're enumerating transition sequences, build graphs
/// from those and then in turn build reconvergent paths from those graphs, it's
/// better to pair them like this instead of playing around with indices.
struct ReconvergentPathWithGraph {
  ReconvergentPath path;
  const ReconvergentPathFinderGraph *graph;

  ReconvergentPathWithGraph(ReconvergentPath p,
                            const ReconvergentPathFinderGraph *g)
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
  ArrayRef<ReconvergentPathWithGraph> reconvergentPaths;
  ArrayRef<SynchronizingCyclePair> syncCyclePairs;

  /// Reference to synchronizing cycles graph.
  const SynchronizingCyclesFinderGraph &syncGraph;

  /// CFDFCs needed for cylce constraints.
  ArrayRef<CFDFC *> cfdfcs;

  /// Computed minimum feasible Initiation Interval across all CFDFCs (set by addCycleTimeConstraints).
  double computedII = 1.0;

  void addLatencyVariables();

  /// Add pattern imbalance constraints for reconvergent paths.
  void addReconvergentPathConstraints();

  /// Add pattern imbalance constraints for synchronizing cycles.
  void addSyncCycleConstraints();

  void addStallPropagationConstraints();

  /// Add cycle time (II) constraints for each CFDFC cycle.
  void addCycleTimeConstraints();

  /// Minimize stalls first, then latency cost.
  void setLatencyBalancingObjective();

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

  ArrayRef<ReconvergentPathWithGraph> reconvergentPaths;
  ArrayRef<CFDFC *> cfdfcs;

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