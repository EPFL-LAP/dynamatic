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
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
namespace dynamatic {
namespace buffer {

/// Abstract class holding the basic logic for the smart buffer placement pass,
/// which expresses the buffer placement problem in dataflow circuits as an MILP
/// (mixed-integer linear program) whose solution indicates the location and
/// nature of buffers that must be placed in the circuit to achieve functional
/// correctness and high performance. Specific implementations of MILP-based
/// buffer placement algorithms can inherit from this class to benefit from
/// some initialization/state management code they are likely to need. Most
/// public and protected methods are virtual to allow implementors to customize
/// the optimization flow to their nedds. Gurobi's C++ API is used internally to
/// manage the MILP.
class BufferPlacementMILP {
public:
  /// Denotes the internal state of the MILP.
  enum class MILPStatus {
    /// Failed to setup the MILP, it cannot be optimized.
    FAILED_TO_SETUP,
    /// Requested buffering properties are unsatisfiable.
    UNSAT_PROPERTIES,
    /// The MILP is ready to be optimized.
    READY,
    /// MILP optimization failed, placement results cannot be extracted.
    FAILED_TO_OPTIMIZE,
    /// MILP optimization succeeded, placement results can be extracted.
    OPTIMIZED
  };

  /// Contains timing characterizations for dataflow components required to
  /// create the MILP constraints.
  const TimingDatabase &timingDB;

  /// Initializes the buffer placement MILP for a Handshake function (with its
  /// CFDFCs) with specific compoennt timing models. The Gurobi environment is
  /// used to create the internal Gurobi model. If a logger is passed, the
  /// object may log information to it during its lifetime. The lifetime of
  /// passed references must exceed the object's lifetime, which doesn't copy
  /// the underlying data. The constructor fills the `channels` map with a
  /// mapping between each of the function's channel and their specific
  /// buffering properties, adjusting for components' internal buffers given by
  /// the timing models. If some buffering properties become unsatisfiable
  /// following this step, the constructor sets the MILP status to
  /// `MILPStatus::UNSAT_PROPERTIES` before returning, otherwise the status is
  /// `MILPStatus::FAILED_TO_SETUP` when the constructor returns.
  BufferPlacementMILP(FuncInfo &funcInfo, const TimingDatabase &timingDB,
                      GRBEnv &env, Logger *log = nullptr);

  /// Optimizes the MILP. If a logger was provided at object creation, the MILP
  /// model and its solution are stored in plain text in its associated
  /// directory. If a valid pointer is provided, saves Gurobi's optimization
  /// status code in it after optimization.
  virtual LogicalResult optimize(int *milpStat = nullptr);

  /// Fills in the provided map with the buffer placement results after MILP
  /// optimization, specifying how each channel (equivalently, each MLIR value)
  /// must be bufferized according to the MILP solution.
  virtual LogicalResult
  getPlacement(DenseMap<Value, PlacementResult> &placement) = 0;

  /// Returns the MILP's status.
  MILPStatus getStatus() { return status; }

  /// Determines whether the MILP is in a valid state to be optimized. If this
  /// returns true, optimize can be called to solve the MILP. Conversely, if
  /// this returns false then a call to optimize will produce an error.
  bool isReadyForOptimization() { return status == MILPStatus::READY; };

  /// The class manages a Gurobi model which should not be copied, hence the
  /// copy constructor is deleted.
  BufferPlacementMILP(const BufferPlacementMILP &) = delete;

  /// The class manages a Gurobi model which should not be copied, hence the
  /// copy-assignment constructor is deleted.
  BufferPlacementMILP &operator=(const BufferPlacementMILP &) = delete;

  /// Virtual default destructor.
  virtual ~BufferPlacementMILP() = default;

protected:
  /// Aggregates all data members related to the Handshake function under
  /// optimization.
  FuncInfo &funcInfo;
  /// After construction, maps all channels (i.e, values) defined in the
  /// function to their specific channel buffering properties (unconstraining
  /// properties if none were explicitly specified).
  llvm::MapVector<Value, ChannelBufProps> channels;
  /// Gurobi model for creating/solving the MILP.
  GRBModel model;
  /// Logger; if not null the class will log setup and results information.
  Logger *logger;
  /// MILP's status, which changes during the object's lifetime.
  MILPStatus status = MILPStatus::FAILED_TO_SETUP;

  /// Adds pre-existing buffers that may exist as part of the units the channel
  /// connects to to the buffering properties. These are added to the minimum
  /// numbers of transparent and opaque slots so that the MILP is forced to
  /// place at least a certain quantity of slots on the channel and can take
  /// them into account in its constraints. Fails when buffering properties
  /// become unsatisfiable due to an increase in the minimum number of slots;
  /// succeeds otherwise.
  virtual LogicalResult addInternalBuffers(Channel &channel);

  /// Removes pre-existing buffers that may exist as part of the units the
  /// channel connects to from the placement results. These are deducted from
  /// the numbers of transparent and opaque slots stored in the placement
  /// results. The latter are expected to specify more slots than what is going
  /// to be deducted (which should be guaranteed by the MILP constraints).
  virtual void deductInternalBuffers(Channel &channel, PlacementResult &result);

  /// Helper method to run a callback function on each input/output port pair of
  /// the provided operation, unless one of the ports has `mlir::MemRefType`.
  virtual void forEachIOPair(Operation *op,
                             const std::function<void(Value, Value)> &callback);
};

} // namespace buffer
} // namespace dynamatic

/// Prints a description of the buffer placement MILP's current status to an
/// output stream.
template <typename T>
T &operator<<(T &os,
              dynamatic::buffer::BufferPlacementMILP::MILPStatus &status) {
  switch (status) {
  case dynamatic::buffer::BufferPlacementMILP::MILPStatus::FAILED_TO_SETUP:
    os << "something went wrong during the creation of MILP constraints or "
          "objective, or no objective was set";
    break;
  case dynamatic::buffer::BufferPlacementMILP::MILPStatus::UNSAT_PROPERTIES:
    os << "the custom buffer placement constraints derived from custom channel "
          "buffering properties attached to IR operations are unsatisfiable "
          "with respect to the provided compoennt models";
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
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERPLACEMENTMILP_H
