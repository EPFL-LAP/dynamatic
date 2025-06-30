//===- FPL22Buffers.h - FPL'22 buffer placement -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FPL'22 smart buffer placement, as presented in
// https://ieeexplore.ieee.org/abstract/document/10035122
//
// This file declares the `FPL22Placement` class, which inherits the abstract
// `BufferPlacementMILP` class to setup and solve a real MILP from which
// buffering decisions can be made. Every public member declared in this file is
// under the `dynamatic::buffer::fpl22` namespace, as to not create name
// conflicts for common structs with other implementors of
// `BufferPlacementMILP`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPL22BUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPL22BUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "llvm/ADT/MapVector.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace dynamatic {
namespace buffer {
namespace fpl22 {

/// Holds common API for all MILPs involved in FPL'22 buffer placement.
class FPL22BuffersBase : public BufferPlacementMILP {
protected:
  /// Just forwards its arguments to the super class constructor with the same
  /// signature.
  FPL22BuffersBase(GRBEnv &env, FuncInfo &funcInfo,
                   const TimingDatabase &timingDB, double targetPeriod)
      : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod){};

  /// Just forwards its arguments to the super class constructor with the same
  /// signature.
  FPL22BuffersBase(GRBEnv &env, FuncInfo &funcInfo,
                   const TimingDatabase &timingDB, double targetPeriod,
                   Logger &logger, StringRef milpName)
      : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                            milpName){};

  /// Interprets the MILP solution to derive buffer placement decisions. Since
  /// the MILP cannot encode the placement of both opaque and transparent slots
  /// on a single channel, some "interpretation" of the results is necessary to
  /// derive "mixed" placements where some buffer slots are opaque and some are
  /// transparent.
  void extractResult(BufferPlacement &placement) override;

  /// Adds channel-specific buffering constraints that were parsed from IR
  /// annotations to the Gurobi model.
  void addCustomChannelConstraints(Value channel);

  /// Adds path constraints between pins of the unit's input and output ports in
  /// different timing domains. At the moment the set of mixed-domain
  /// constraints is determined by the method itself, which contains
  /// case-by-case logic to derive them based on the operation's type.
  /// Eventually, we will support formal timing models that will themselves
  /// carry that information, at which point this method will be drastically
  /// simplified.
  ///
  /// A `filter` can be provided to filter out constraints involving input or
  /// output ports connected to channels for which the filter returns false. The
  /// default filter always returns true. It is only valid to call this method
  /// after having added channel variables to the model for all channels
  /// adjacent to the unit, unless these channels are filtered out by the
  /// `filter` function.
  void addUnitMixedPathConstraints(Operation *unit,
                                   ChannelFilter filter = nullFilter);
};

/// This MILP operates on the channels and units from a single CFDFC union
/// derived from the set of CFDFCs identified for a Handshake function. It takes
/// into account all timing domains of dataflow circuits (data, valid, ready) to
/// derive an optimal buffer placement. It creates
/// 1. custom channel constraints derived from channel-specific buffering
///    properties
/// 2. path constraints on all timing domains (including mixed-domain
///    connections within units)
/// 3. elasticity constraints
/// 4. throughput constraints for all CFDFCs that are part of the CFDFC union
/// 5. a maximixation objective, that rewards high CFDFC throughputs and
///    penalizes the placement of many large buffers in the circuit
class CFDFCUnionBuffers : public FPL22BuffersBase {
public:
  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. If a channel's buffering properties are provably
  /// unsatisfiable, the MILP will not be marked ready for optimization,
  /// ensuring that further calls to `optimize` fail.
  CFDFCUnionBuffers(GRBEnv &env, FuncInfo &funcInfo,
                    const TimingDatabase &timingDB, double targetPeriod,
                    CFDFCUnion &cfUnion);

  /// Achieves the same as the other constructor but additionally logs placement
  /// decisions and achieved throughputs using the provided logger, and dumps
  /// the MILP model and solution at the provided name next to the log file.
  CFDFCUnionBuffers(GRBEnv &env, FuncInfo &funcInfo,
                    const TimingDatabase &timingDB, double targetPeriod,
                    CFDFCUnion &cfUnion, Logger &logger, StringRef milpName);

private:
  /// The CFDFC union over which the MILP is described. Constraints are only
  /// created over the channels and units that are part of this union.
  CFDFCUnion &cfUnion;

  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

/// This MILP operates on the channels and units that are outside of all CFDFCs
/// identified in a Handshake function. It takes into account all timing domains
/// of dataflow circuits (data, valid, ready) to derive an optimal buffer
/// placement. It creates
/// 1. custom channel constraints derived from channel-specific buffering
///    properties
/// 2. path constraints on all timing domains (including mixed-domain
///    connections within units)
/// 3. elasticity constraints
/// 5. a maximixation objective that penalizes the placement of many large
///    buffers in the circuit
class OutOfCycleBuffers : public FPL22BuffersBase {
public:
  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. If a channel's buffering properties are provably
  /// unsatisfiable, the MILP will not be marked ready for optimization,
  /// ensuring that further calls to `optimize` fail.
  OutOfCycleBuffers(GRBEnv &env, FuncInfo &funcInfo,
                    const TimingDatabase &timingDB, double targetPeriod);

  /// Achieves the same as the other constructor but additionally logs placement
  /// decisions and achieved throughputs using the provided logger, and dumps
  /// the MILP model and solution at the provided name next to the log file.
  OutOfCycleBuffers(GRBEnv &env, FuncInfo &funcInfo,
                    const TimingDatabase &timingDB, double targetPeriod,
                    Logger &logger, StringRef milpName = "out_of_cycle");

private:
  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

} // namespace fpl22
} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPL22BUFFERS_H