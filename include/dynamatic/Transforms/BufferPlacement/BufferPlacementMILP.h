//===- BufferPlacementMILP.h - MILP-based buffer placement ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common infrastructure for MILP-based buffer placement (requires Gurobi). This
// mainly declares the abstract `BufferPlacementMILP` class, which contains some
// common logic to manage an MILP that represents a buffer placement problem.
// Buffer placements algorithms should subclass it to get some of the common
// boilerplate code they are likely to need for free.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERPLACEMENTMILP_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERPLACEMENTMILP_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/MILP.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
namespace dynamatic {
namespace buffer {

/// Pair of Gurobi variables meant to represent the arrival times of a signal at
/// a channel's endpoints.
struct TimeVars {
  /// Time at channel's input (i.e., at source unit's output port).
  GRBVar tIn;
  /// Time at channel's output (i.e., at destination unit's input port).
  GRBVar tOut;
};

/// Holds MILP variables associated to every CFDFC unit. Note that a unit may
/// appear in multiple CFDFCs and so may have multiple sets of these variables.
struct UnitVars {
  /// Fluid retiming of tokens at unit's input (real).
  GRBVar retIn;
  /// Fluid retiming of tokens at unit's output. Identical to retiming at unit's
  /// input if the latter is combinational (real).
  GRBVar retOut;
};

/// Holds MILP variables related to a specific signal (e.g., data, valid, ready)
/// of a dataflow channel.
struct ChannelSignalVars {
  /// Arrival time of the signal at channel's endpoints.
  TimeVars path;
  /// Presence of a buffer on the signal.
  GRBVar bufPresent;
};

/// Holds all MILP variables associated to a channel.
struct ChannelVars {
  /// For specific signals on the channel, arrival time at channel's endpoints
  /// (real, real) and buffer presence (binary).
  std::map<SignalType, ChannelSignalVars> signalVars;
  /// Elastic arrival time at channel's endpoints (real).
  TimeVars elastic;
  /// Presence of any buffer on the channel (binary).
  GRBVar bufPresent;
  /// Number of buffer slots on the channel (integer).
  GRBVar bufNumSlots;
};

/// Holds all variables associated to a CFDFC. These are a set of variables for
/// each unit inside the CFDFC, a throughput variable for each channel inside
/// the CFDFC, and a CFDFC throughput variable.
struct CFDFCVars {
  /// Maps each CFDFC unit to its retiming variables.
  llvm::MapVector<Operation *, UnitVars> unitVars;
  /// Channel throughput variables (real).
  llvm::MapVector<Value, GRBVar> channelThroughputs;
  /// CFDFC throughput (real).
  GRBVar throughput;
};

/// Holds all variables that may be used in the MILP. These are a set of
/// variables for each CFDFC and a set of variables for each channel in the
/// function.
struct MILPVars {
  /// Mapping between CFDFCs and their related variables.
  llvm::MapVector<CFDFC *, CFDFCVars> cfVars;
  /// Mapping between channels and their related variables.
  llvm::MapVector<Value, ChannelVars> channelVars;
};

/// Abstract class holding the basic logic for the smart buffer placement pass,
/// which expresses the buffer placement problem in dataflow circuits as an MILP
/// (mixed-integer linear program) whose solution indicates the location and
/// nature of buffers that must be placed in the circuit to achieve functional
/// correctness and high performance. Specific implementations of MILP-based
/// buffer placement algorithms can inherit from this class to benefit from
/// some pre/post-processind steps and verification they are likely to need.
class BufferPlacementMILP : public MILP<BufferPlacement> {
public:
  /// Contains timing characterizations for dataflow components required to
  /// create the MILP constraints.
  const TimingDatabase &timingDB;
  /// Target clock period.
  const double targetPeriod;

  /// Starts setting up a the buffer placement MILP for a Handshake function
  /// (with its CFDFCs) with specific component timing models. The constructor
  /// maps each of the function's channel to its specific buffering properties,
  /// adjusting for components' internal buffers given by the timing models. If
  /// some buffering properties become unsatisfiable following this step, the
  /// constructor sets the `unsatisfiable` flag to true.
  BufferPlacementMILP(GRBEnv &env, FuncInfo &funcInfo,
                      const TimingDatabase &timingDB, double targetPeriod);

  /// Follows the same pre-processing step as the other constructor; in
  /// addition, dumps the MILP model and solution under the provided name in the
  /// logger's directory.
  BufferPlacementMILP(GRBEnv &env, FuncInfo &funcInfo,
                      const TimingDatabase &timingDB, double targetPeriod,
                      Logger &logger, StringRef milpName);

protected:
  /// For unit constraints, oracle function determining whether constraints
  /// corresponding to the port should be added to the MILP model.
  using ChannelFilter = const std::function<bool(Value)> &;
  /// Default channel filter for unit constraints, which doesn't filter out any
  /// channel.
  static bool nullFilter(Value channel) { return true; }

  /// Aggregates all data members related to the Handshake function under
  /// optimization.
  FuncInfo &funcInfo;
  /// After construction, maps all channels (i.e, values) defined in the
  /// function to their specific channel buffering properties (unconstraining
  /// properties if none were explicitly specified).
  llvm::MapVector<Value, ChannelBufProps> channelProps;
  /// Logger; if not null the class may log setup and result information to it.
  Logger *logger;
  /// Contains all variables used throughout the MILP. Variables can be added to
  /// it with the `BufferPlacementMILP::addChannelVars` and
  /// `BufferPlacementMILP::addCFDFCVars` methods.
  MILPVars vars;

  /// Whether the MILP was determined to be unsatisfiable during construction.
  bool unsatisfiable = false;

  /// Adds channel variables to the MILP model for the provided channel.
  /// Signal-specific variables will be added for the provided signal types
  /// only.
  void addChannelVars(Value channel, ArrayRef<SignalType> signals);

  /// Adds CFDFC variables to the MILP model for the provided CFDFC. These are
  /// a pair of retiming variables for each CFDFC unit, a throughput variable
  /// for each CFDFC channel, and an overall CFDFC's throughput variable.
  void addCFDFCVars(CFDFC &cfdfc);

  /// Adds path constraints for a specific signal type between the unit's input
  /// and output ports. If the internal path for the signal is combinational, a
  /// constraint is added for every input/output port pair. Otherwise, a
  /// constraint is added for each individual input port and output port.
  ///
  /// A `filter` can be provided to filter out constraints involving input or
  /// output ports connected to channels for which the filter returns false. The
  /// default filter always returns true. It is only valid to call this method
  /// after having added channel variables to the model for all channels
  /// adjacent to the unit, unless these channels are filtered out by the
  /// `filter` function.
  void addUnitPathConstraints(Operation *unit, SignalType type,
                              ChannelFilter filter = nullFilter);

  /// Adds elasticity constraints for the channel. The `signalGroups` argument
  /// should contain all the signal types with which channel variables for the
  /// specific channel were added exactly once. Furthermore, these signals
  /// can be separated in multiple groups to force the MILP to place buffers for
  /// all signals within each group at the same locations. For example, if one
  /// can only place two buffer types, one which cuts both the data and valid
  /// signals and one which cuts the ready signal only, and channel variables
  /// were created for all those signals, then `signalGroups` should be:
  /// [ [SignalType::DATA, SignalType::VALID],
  ///   [SignalType::READY]                   ]
  /// The order within each group is irrelevant; the resulting constraints will
  /// be identical modulo a reordering of the terms.
  void
  addChannelElasticityConstraints(Value channel,
                                  ArrayRef<ArrayRef<SignalType>> signalGroups);

  /// Adds elasticity constraints between the unit's input and output ports. A
  /// constraint is added for every input/output port pair.
  ///
  /// A `filter` can be provided to filter out constraints involving input or
  /// output ports connected to channels for which the filter returns false. The
  /// default filter always returns true. It is only valid to call this method
  /// after having added channel variables to the model for all channels
  /// adjacent to the unit, unless these channels are filtered out by the
  /// `filter` function.
  void addUnitElasticityConstraints(Operation *unit,
                                    ChannelFilter filter = nullFilter);

  /// Adds throughput constraints for all channels in the CFDFC. Throughput is a
  /// data-centric notion, so it only makes sense to call this method if channel
  /// variables were created for the data signal.
  ///
  /// It is only valid to call this method after having added variables for the
  /// CFDFC and variables for the data signal of all channels inside the CFDFC
  /// to the model.
  void addChannelThroughputConstraints(CFDFC &cfdfc);

  /// Adds throughput constraints for all units in the CFDFC. A single
  /// constraint is added for all units with non-zero latency on their datapath.
  ///
  /// It is only valid to call this method after having added variables for the
  /// CFDFC to the model.
  void addUnitThroughputConstraints(CFDFC &cfdfc);

  /// Returns an estimation of the number of times a token will be transfered on
  /// the input channel. The estimation is based on the Handshake function's
  /// extracted CFDFCs.
  unsigned getChannelNumExecs(Value channel);

  /// Adds the MILP model's objective to maximize. The objective has a positive
  /// "throughput term" for every provided CFDFC. These terms are weighted by
  /// the "importance" of the CFDFC compared to the others, which is determined
  /// using an estimation of the transfer frequency over each provided channel.
  /// The objective has a negative term for each buffer placement decision and
  /// for each buffer slot placed on any of the provide channels.
  void addObjective(ValueRange channels, ArrayRef<CFDFC *> cfdfcs);

  /// Adds pre-existing buffers that may exist as part of the units the channel
  /// connects to to the buffering properties. These are added to the minimum
  /// numbers of transparent and opaque slots so that the MILP is forced to
  /// place at least a certain quantity of slots on the channel and can take
  /// them into account in its constraints.
  void addInternalBuffers(Channel &channel);

  /// Removes pre-existing buffers that may exist as part of the units the
  /// channel connects to from the placement results. These are deducted from
  /// the numbers of transparent and opaque slots stored in the placement
  /// results. The latter are expected to specify more slots than what is going
  /// to be deducted (which should be guaranteed by the MILP constraints).
  void deductInternalBuffers(Value channel, PlacementResult &result);

  /// Helper method to run a callback function on each input/output port pair of
  /// the provided operation, unless one of the ports has `mlir::MemRefType`.
  void forEachIOPair(Operation *op,
                     const std::function<void(Value, Value)> &callback);

  /// Logs placement decisisons and achieved throughputs after MILP
  /// optimization. Asserts if the logger is nullptr.
  void logResults(DenseMap<Value, PlacementResult> &placement);

private:
  /// Large constant strictly greater than the number of units in the Handshake
  /// function under consideration.
  unsigned largeCst;

  /// During object construction, map all the function's channels to their
  /// specific buffering properties, adjusting for buffers within units as
  /// described by the timing models. Fails if the buffering properties of a
  /// channel are unsatisfiable or become unsatisfiable after adjustment.
  LogicalResult mapChannelsToProperties();
};

} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERPLACEMENTMILP_H
