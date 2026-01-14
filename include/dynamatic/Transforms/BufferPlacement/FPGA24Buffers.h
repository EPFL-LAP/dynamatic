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
namespace fpl24 {

/// Latency Balancing MILP ///

/// Holds the result of the first LP for usage in the LP.
struct LatencyBalancingResult {
  /// Map from channel to its computed extra latency.
  DenseMap<Value, unsigned> channelExtraLatency;
  /// Target intiation interval.
  double targetII;
};

class LatencyBalancingMILP : public BufferPlacementMILP {
public:
  LatencyBalancingMILP(CPSolver::SolverKind solverKind, int timeout,
                       FuncInfo &funcInfo, const TimingDatabase &timingDB,
                       double targetPeriod,
                       ArrayRef<ReconvergentPath> reconvergentPaths,
                       ArrayRef<SynchronizingCyclePair> syncCyclePairs,
                       const ReconvergentPathFinderGraph &reconvergentGraph,
                       const SynchronizingCyclesFinderGraph &syncGraph,
                       ArrayRef<CFDFC *> cfdfcs);

  /// Extract latency results after solving.
  LatencyBalancingResult extractLatencyResults();

protected:
  /// Interpret solution - not used for buffer placement directly.
  void extractResult(BufferPlacement &placement) override;

private:
  ArrayRef<ReconvergentPath> reconvergentPaths;
  ArrayRef<SynchronizingCyclePair> syncCyclePairs;

  /// References to the graphs for accessing node/edge information.
  const ReconvergentPathFinderGraph &reconvergentGraph;
  const SynchronizingCyclesFinderGraph &syncGraph;

  /// CFDFCs needed for cylce constraints.
  ArrayRef<CFDFC *> cfdfcs;

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

/// Occupancy Balancing MILP ///

class OccupancyBalancingMILP : public BufferPlacementMILP {
public:
  OccupancyBalancingMILP(CPSolver::SolverKind solverKind, int timeout,
                         FuncInfo &funcInfo, const TimingDatabase &timingDB,
                         double targetPeriod,
                         const LatencyBalancingResult &latencyResult,
                         ArrayRef<ReconvergentPath> reconvergentPaths,
                         ArrayRef<SynchronizingCyclePair> syncCyclePairs,
                         const ReconvergentPathFinderGraph &reconvergentGraph,
                         const SynchronizingCyclesFinderGraph &syncGraph,
                         ArrayRef<CFDFC *> cfdfcs);

protected:
  /// Extract buffer placement decisions from the solved MILP.
  void extractResult(BufferPlacement &placement) override;

private:
  /// Results from LP1
  const LatencyBalancingResult &latencyResult;

  ArrayRef<ReconvergentPath> reconvergentPaths;
  ArrayRef<SynchronizingCyclePair> syncCyclePairs;

  /// References to the graphs for accessing node/edge information.
  const ReconvergentPathFinderGraph &reconvergentGraph;
  const SynchronizingCyclesFinderGraph &syncGraph;

  /// CFDFCs
  ArrayRef<CFDFC *> cfdfcs;

  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();

  void addOccupancyVariables();

  void addMinimumOccupancyConstraints();

  void addPathConsistencyConstraints();

  void addCycleCapacityConstraints();

  /// Minimize total buffer area.
  void setOccupancyObjective();
};

class FPGA24Buffers : public BufferPlacementMILP {
protected:
  /// TODO: Explain
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

} // namespace fpl24
} // namespace buffer
} // namespace dynamatic

#endif /// DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPGA24BUFFERS_H