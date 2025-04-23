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
  llvm::MapVector<CFDFC *, CFDFCVars> cfdfcVars;
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
  /// Represents a list of signals that are buffered together by a single
  /// buffer type, which is denoted by its (potentially null) timing model.
  struct BufferingGroup {
    /// List of signals buffered by the specific buffer. This must contain at
    /// least one signal type. The first signal in the list is considered the
    /// "reference" for this group. The ordering of these signals should only
    /// change the MILP cosmetically.
    SmallVector<SignalType> signals;
    /// Buffer's timing model.
    const TimingModel *bufModel;

    /// Simple member-by-member constructor. At least one signal must be
    /// provided, otherwise the cosntructor will assert.
    BufferingGroup(ArrayRef<SignalType> signals, const TimingModel *bufModel)
        : signals(signals), bufModel(bufModel) {
      assert(!signals.empty() && "list of signals cannot be empty");
    }

    /// Returns the reference signals of the group.
    SignalType getRefSignal() const { return signals.front(); };
    /// Returns the "other" signals in the group i.e., those that are not the
    /// reference. The returned array is empty if the group only contains one
    /// signal.
    ArrayRef<SignalType> getOtherSignals() const {
      return ArrayRef<SignalType>(signals).drop_front();
    };

    /// Returns the combinational delay of the group's buffer for a signal type.
    double getCombinationalDelay(Value channel, SignalType signal) const;
  };

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
  llvm::MapVector<Value, handshake::ChannelBufProps> channelProps;
  /// Logger; if not null the class may log setup and result information to it.
  Logger *logger;
  /// Contains all variables used throughout the MILP. Variables can be added to
  /// it with the `BufferPlacementMILP::addChannelVars` and
  /// `BufferPlacementMILP::addCFDFCVars` methods.
  MILPVars vars;

  /// Whether the MILP was determined to be unsatisfiable during construction.
  bool unsatisfiable = false;
  /// Large constant strictly greater than the number of units in the Handshake
  /// function under consideration.
  unsigned largeCst;

  /// Adds channel variables to the MILP model for the provided channel.
  /// Signal-specific variables will be added for the provided signal types
  /// only.
  void addChannelVars(Value channel, ArrayRef<SignalType> signals);

  /// Adds CFDFC variables to the MILP model for the provided CFDFC. These are
  /// a pair of retiming variables for each CFDFC unit, a throughput variable
  /// for each CFDFC channel, and an overall CFDFC's throughput variable.
  void addCFDFCVars(CFDFC &cfdfc);

  /// Adds buffer presence constraints for the provided signals on the channel. 
  /// These ensure buffer presence variables are properly linked with signal 
  /// latency and channel buffer presence.
  void addSimpleBufferPresenceConstraints(Value channel, ArrayRef<SignalType> signals);

  /// Adds constraints that ensure the arrival times at both ends of the 
  /// channel are less than or equal to the target clock period.
  void addTargetPeriodConstraints(Value channel, ArrayRef<SignalType> signals);
  
  /// Adds simple delay propagation constraints for the channel. These ensure 
  /// proper signal propagation along the channel.
  ///
  /// Choose only one function between 'addSimpleChannelTimingConstraints' 
  /// and 'addBufferTimingConstraints'.
  void addSimpleChannelTimingConstraints(Value channel, ArrayRef<SignalType> signals);

  /// Adds constraints for buffer delay propagation on the channel. These account
  /// for delays introduced by buffers on the signal paths.
  ///
  /// Choose only one function between 'addSimpleChannelTimingConstraints' 
  /// and 'addBufferTimingConstraints'.
  void addBufferTimingConstraints(Value channel, SignalType signal,
                                 const TimingModel *bufModel,
                                 ArrayRef<BufferingGroup> before = {},
                                 ArrayRef<BufferingGroup> after = {});

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
  void addUnitTimingConstraints(Operation *unit, SignalType signal,
                              ChannelFilter filter = nullFilter);

  /// Adds elasticity constraints for the channel. The buffering groups should
  /// contain all the signal types with which channel variables for the specific
  /// channel were added exactly once. Groups force the MILP to place buffers
  /// for all signals within each group at the same locations.
  void addChannelElasticityConstraints(Value channel,
                                       ArrayRef<BufferingGroup> bufGroups);

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

  /// Adds constraints for token distribution across the CFDFC. These ensure
  /// proper token flow based on fluid retiming variables.
  ///
  /// It is only valid to call this method after having added variables for the
  /// CFDFC to the model.
  void addTokenDistributionConstraints(CFDFC &cfdfc);

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

  /// Helper method to run a callback function on each input/output port pair of
  /// the provided operation, unless one of the ports has `mlir::MemRefType`.
  void forEachIOPair(Operation *op,
                     const std::function<void(Value, Value)> &callback);

  /// Logs placement decisisons and achieved throughputs after MILP
  /// optimization. Asserts if the logger is nullptr.
  void logResults(BufferPlacement &placement);

private:
  /// Common logic for all constructors. Fills the channel to buffering
  /// properties mapping and defines a large constant used for elasticity
  /// constraints.
  void initialize();
};

} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERPLACEMENTMILP_H
