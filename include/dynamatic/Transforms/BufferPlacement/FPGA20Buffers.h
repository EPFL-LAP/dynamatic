//===- FPGA20Buffers.h - FPGA'20 buffer placement ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FPGA'20 smart buffer placement, as presented in
// https://dl.acm.org/doi/full/10.1145/3477053
//
// This mainly declares the `FPGA20Placement` class, which inherits the abstract
// `BufferPlacementMILP` class to setup and solve a real MILP from which
// buffering decisions can be made. Every public member declared in this file is
// under the `dynamatic::buffer::fpga20` namespace, as to not create name
// conflicts for common structs with other implementors of
// `BufferPlacementMILP`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPGA20PLACEMENT_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPGA20PLACEMENT_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace dynamatic {
namespace buffer {
namespace fpga20 {
/// Holds MILP variables associated to every CFDFC unit. Note that a unit may
/// appear in multiple CFDFCs and so may have multiple sets of these variables.
struct UnitVars {
  /// Fluid retiming of tokens at unit's input (real).
  GRBVar retIn;
  /// Fluid retiming of tokens at unit's output. Identical to retiming at unit's
  /// input if the latter is combinational (real).
  GRBVar retOut;
};

/// Holds all MILP variables associated to a channel.
struct ChannelVars {
  /// Arrival time at channel's input (real).
  GRBVar tPathIn;
  /// Arrival time at channel's output (real).
  GRBVar tPathOut;
  /// Elastic arrival time at channel's input (real).
  GRBVar tElasIn;
  /// Elastic arrival time at channel's output (real).
  GRBVar tElasOut;
  /// Whether there is a buffer of any kind on the channel (binary).
  GRBVar bufPresent;
  /// Whether the buffer on the channel is opaque (binary).
  GRBVar bufIsOpaque;
  /// Number of buffer slots on the channel (integer).
  GRBVar bufNumSlots;
};

/// Holds all variables associated to a CFDFC. These are a set of variables for
/// each unit inside the CFDFC, a throughput variable for each channel inside
/// the CFDFC, and a CFDFC throughput varriable.
struct CFDFCVars {
  /// Maps each CFDFC unit to its retiming variables.
  llvm::MapVector<Operation *, UnitVars> units;
  /// Channel throughput variables  (real).
  llvm::MapVector<Value, GRBVar> channelThroughputs;
  /// CFDFC throughput (real).
  GRBVar throughput;
};

/// Holds all variables that may be used in the MILP. These are a set of
/// variables for each CFDFC and a set of variables for each channel in the
/// function.
struct MILPVars {
  /// Mapping between each CFDFC and their related variables.
  llvm::MapVector<CFDFC *, CFDFCVars> cfdfcs;
  /// Mapping between each circuit channel and their related variables.
  llvm::MapVector<Value, ChannelVars> channels;
};

/// Holds the state and logic for FPGA'20 smart buffer placement. To buffer a
/// dataflow circuit, this MILP-based algorithm creates:
/// 1. custom channel constrants derived from channel-specific buffering
///    properties
/// 2. path constraints for all non-memory channels and units
/// 3. elasticity constraints for all non-memory channels and units
/// 4. throughput constraints for all channels and units parts of CFDFCs that
///    were extracted from the function
/// 5. a maximixation objective, that rewards high CFDFC throughputs and
///    penalizes the placement of many large buffers in the circuit
class FPGA20Buffers : public BufferPlacementMILP {
public:
  /// Target clock period.
  const double targetPeriod;
  /// Maximum clock period.
  const double maxPeriod;
  /// Whether to use the same placement policy as legacy Dynamatic; non-legacy
  /// placement will yield faster circuits (some opaque slots transformed into
  /// transparent slots).
  const bool legacyPlacement;

  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. The `legacyPlacemnt` controls the interpretation of the
  /// MILP's results (non-legacy placement should yield faster circuits in
  /// general). If a channel's buffering properties are provably unsatisfiable,
  /// the MILP status will be set to `MILPStatus::UNSAT_PROPERTIES` before
  /// returning. If something went wrong during MILP setup, the MILP status will
  /// be set to `MILPStatus::FAILED_TO_SETUP`.
  FPGA20Buffers(FuncInfo &funcInfo, const TimingDatabase &timingDB, GRBEnv &env,
                Logger *log = nullptr, double targetPeriod = 4.0,
                double maxPeriod = 8.0, bool legacyPlacement = true);

  /// Interprets the MILP solution to derive buffer placement decisions. Since
  /// the MILP cannot encode the placement of both opaque and transparent slots
  /// on a single channel, some "interpretation" of the results is necessary to
  /// derive "mixed" placements where some buffer slots are opaque and some are
  /// transparent. This interpretation is partically controlled by the
  /// `legacyPlacement` flag, and always respects the channel-specific buffering
  /// constraints.
  LogicalResult
  getPlacement(DenseMap<Value, PlacementResult> &placement) override;

protected:
  /// Contains all variables used throughout the MILP.
  MILPVars vars;

  /// Setups the entire MILP, first creating all variables, then all
  /// constraints, and finally setting the system's objective. Called by the
  /// constructor in the absence of prior failures, after which the MILP is
  /// ready to be optimized.
  LogicalResult setup();

  /// Adds all variables used in the MILP to the Gurobi model.
  LogicalResult createVars();

  /// Adds all variables related to the passed CFDFC to the Gurobi model. Each
  /// time this method is called, it must be with a different uid which is used
  /// to unique the name of each created variable. The CFDFC must be part of
  /// those that were provided to the constructor.
  LogicalResult createCFDFCVars(CFDFC &cfdfc, unsigned uid);

  /// Adds all variables related to all channels (regardless of whether they are
  /// part of a CFDFC) to the Gurobi model.
  LogicalResult createChannelVars();

  /// Adds channel-specific buffering constraints that were parsed from IR
  /// annotations to the Gurobi model.
  LogicalResult addCustomChannelConstraints(ValueRange customChannels);

  /// Adds path constraints for all provided channels and units to the Gurobi
  /// model. All channels and units must be part of the Handshake function under
  /// consideration.
  LogicalResult addPathConstraints(ValueRange pathChannels,
                                   ArrayRef<Operation *> pathUnits);

  /// Adds elasticity constraints for all provided channels and units to the
  /// Gurobi model. All channels and units must be part of the Handshake
  /// function under consideration.
  LogicalResult addElasticityConstraints(ValueRange elasticChannels,
                                         ArrayRef<Operation *> elasticUnits);

  /// Adds throughput constraints for the provided CFDFC to the Gurobi model.
  /// The CFDFC must be part of those that were provided to the constructor.
  LogicalResult addThroughputConstraints(CFDFC &cfdfc);

  /// Adds the objective to the Gurobi model.
  LogicalResult addObjective();

  /// Returns an estimation of the number of times a token will traverse the
  /// input channel. The estimation is based on the extracted CFDFCs.
  unsigned getChannelNumExecs(Value channel);

  /// Logs placement decisisons and achieved throuhgputs after MILP
  /// optimization. Asserts if the logger is nullptr.
  void logResults(DenseMap<Value, PlacementResult> &placement);
};

} // namespace fpga20
} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPGA20PLACEMENT_H