//===- BufferPlacementMILP.h - MILP-based buffer placement ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infrastructure for MILP-based buffer placement (requires Gurobi). This mainly
// declares the BufferPlacementMILP class, which contains all the logic to
// setup, optimize, and extract placement decisions from an MILP that represents
// a buffer placement problem. The class is easily extensible, allowing users to
// customize part of the MILP creation/optimization process.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERPLACEMENTMILP_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERPLACEMENTMILP_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
namespace dynamatic {
namespace buffer {

/// Holds information about what type of buffer should be placed on a specific
/// channel.
struct PlacementResult {
  /// The number of transparent buffer slots that should be placed.
  unsigned numTrans = 0;
  /// The number of opaque buffer slots that should be placed.
  unsigned numOpaque = 0;
  /// Whether opaque slots should be placed transparent slots for placement
  /// results that include both.
  bool opaqueBeforeTrans = true;
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

/// Helper datatype for buffer placement. Simply aggregates all the information
/// related to the Handshake function under optimization.
struct FuncInfo {
  /// The Handshake function in which to place buffers.
  circt::handshake::FuncOp funcOp;
  /// The list of archs in the function (i.e., transitions between basic
  /// blocks).
  SmallVector<experimental::ArchBB> archs;
  /// Maps each CFDFC in the function to a boolean indicating whether it should
  /// be optimized.
  llvm::MapVector<CFDFC *, bool> cfdfcs;

  /// Argument-less constructor so that we can use the struct as a value type
  /// for maps.
  FuncInfo() : funcOp(nullptr){};

  /// Constructs an instance from the function it refers to. Other struct
  /// members start empty.
  FuncInfo(circt::handshake::FuncOp funcOp) : funcOp(funcOp){};
};

/// Holds the bulk of the logic for the smart buffer placement pass, which
/// expresses the buffer placement problem in dataflow circuits as an MILP
/// (mixed-integer linear program) whose solution indicates the location and
/// nature of buffers that must be placed in the circuit to achieve functional
/// correctness and high performance. This class relies on the prior
/// identification of all CFDFCs (choice-free dataflow circuits) inside an input
/// dataflow circuit to create throughput constraints and set the MILP's
/// objective to maximize. Gurobi's C++ API is used internally to manage the
/// MILP.
class BufferPlacementMILP {
public:
  enum class MILPStatus {
    UNSAT_PROPERTIES,
    FAILED_TO_SETUP,
    READY,
    FAILED_TO_OPTIMIZE,
    OPTIMIZED
  };

  /// Contains timing characterizations for dataflow components required to
  /// create the MILP constraints.
  const TimingDatabase &timingDB;
  /// Target clock period.
  const double targetPeriod;
  /// Maximum clock period.
  const double maxPeriod;

  /// Constructs the buffer placement MILP. All arguments passed by reference
  /// must outlive the created instance, which maintains reference internally.
  BufferPlacementMILP(FuncInfo &funcInfo, const TimingDatabase &timingDB,
                      double targetPeriod, double maxPeriod, GRBEnv &env,
                      Logger *log = nullptr);

  /// Determines whether the MILP is in a valid state to be optimized. If this
  /// returns true, optimize can be called to solve the MILP. Conversely, if
  /// this returns false then a call to optimize will produce an error.
  bool isReadyForOptimization() { return status == MILPStatus::READY; };

  /// Optimizes the MILP. If a logger was provided at object creation, the MILP
  /// model and its solution are stored in plain text in its associated
  /// directory. If a valid pointer is provided, saves Gurobi's optimization
  /// status code in it after optimization.
  LogicalResult optimize(int *milpStat = nullptr);

  /// Fills in the provided map with the buffer placement results after MILP
  /// optimization, specifying how each channel (equivalently, each MLIR value)
  /// must be bufferized according to the MILP solution. Setting the legacy
  /// placement flag makes this method use the same placement policy as legacy
  /// Dynamatic; non-legacy placement will yield faster circuits (some opaque
  /// slots transformed into transparent slots).
  LogicalResult getPlacement(DenseMap<Value, PlacementResult> &placement,
                             bool legacyPlacement = true);

  /// Returns the MILP's status.
  MILPStatus getStatus() { return status; }

  /// The class manages a Gurobi model which should not be copied, hence the
  /// copy constructor is deleted.
  BufferPlacementMILP(const BufferPlacementMILP &) = delete;

  /// The class manages a Gurobi model which should not be copied, hence the
  /// copy-assignment constructor is deleted.
  BufferPlacementMILP &operator=(const BufferPlacementMILP &) = delete;

protected:
  /// Aggregates all data members that related to the Handshake function under
  /// optimization.
  FuncInfo &funcInfo;
  /// After construction, maps all channels (i.e, values) defined in the
  /// function to their specific channel buffering properties (unconstraining
  /// properties if none were explicitly specified).
  llvm::MapVector<Value, ChannelBufProps> channels;
  /// Gurobi model for creating/solving the MILP.
  GRBModel model;
  /// Contains all the variables used in the MILP.
  MILPVars vars;
  /// Holds a unique name for each operation in the function.
  DenseMap<Operation *, std::string> nameUniquer;
  /// Logger; if not null the class will log setup and results information.
  Logger *logger;
  /// MILP's status, which changes during the object's lifetime.
  MILPStatus status = MILPStatus::READY;

  /// Setups the entire MILP, first creating all variables, the all constraints,
  /// and finally setting the system's objective. Called by the constructor in
  /// the absence of prior failures, after which the MILP is ready to be
  /// optimized.
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

  /// Adds pre-existing buffers that may exist as part of the units the channel
  /// connects to to the buffering properties. These are added to the minimum
  /// numbers of transparent and opaque slots so that the MILP is forced to
  /// place at least a certain quantity of slots on the channel and can take
  /// them into account in its constraints. Fails when buffering properties
  /// become unsatisfiable due to an increase in the minimum number of slots;
  /// succeeds otherwise.
  LogicalResult addInternalBuffers(Channel &channel);

  /// Removes pre-existing buffers that may exist as part of the units the
  /// channel connects to from the placement results. These are deducted from
  /// the numbers of transparent and opaque slots stored in the placement
  /// results. The latter are expected to specify more slots than what is going
  /// to be deducted (which should be guaranteed by the MILP constraints).
  void deductInternalBuffers(Channel &channel, PlacementResult &result);

  /// Returns a unique name for the channel that corresponds to the passed MLIR
  /// Value (useful for uniquely naming MILP variables).
  std::string getChannelName(Value channel);

private:
  /// Helper method to run a closure on each input/output port pair of the
  /// provided operation, unless one of the ports has type `mlir::MemRefType`.
  void forEachIOPair(Operation *op,
                     const std::function<void(Value, Value)> &callback);

  /// Returns an estimation of the number of times a token will traverse the
  /// input channel. The estimation is based on the extracted function's CFDFCs.
  unsigned getChannelNumExecs(Value channel);

  /// Logs placement decisisons and achieved throuhgputs after MILP
  /// optimization. Asserts if the logger is nullptr.
  void logResults(const DenseMap<Value, PlacementResult> &placement);
};

} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

/// Prints a description of the buffer placement MILP's current status to an
/// output stream.
template <typename T>
T &operator<<(T &os,
              dynamatic::buffer::BufferPlacementMILP::MILPStatus &status) {
  switch (status) {
  case dynamatic::buffer::BufferPlacementMILP::MILPStatus::UNSAT_PROPERTIES:
    os << "the custom buffer placement constraints derived from custom channel "
          "buffering properties attached to IR operations are unsatisfiable "
          "with respect to the provided compoennt models";
    break;
  case dynamatic::buffer::BufferPlacementMILP::MILPStatus::FAILED_TO_SETUP:
    os << "something went wrong during the creation of MILP constraints or "
          "objective";
    break;
  case dynamatic::buffer::BufferPlacementMILP::MILPStatus::READY:
    os << "the MILP is ready to be optimized";
    break;
  case dynamatic::buffer::BufferPlacementMILP::MILPStatus::FAILED_TO_OPTIMIZE:
    os << "the MILP failed to be optimized, check Gurobi's return value for "
          "more details on what went wrong";
    break;
  case dynamatic::buffer::BufferPlacementMILP::MILPStatus::OPTIMIZED:
    os << "the MILP was successfully optimized";
    break;
  }
  return os;
}

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERPLACEMENTMILP_H
