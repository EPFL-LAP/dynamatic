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
// This mainly declares the `FPL22Placement` class, which inherits the abstract
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

/// Temporarily used for channel path constraints. Denotes the potential
/// presence of a buffer type that doens't cut the current signal under
/// consideration but may add a combinational delay to the channel. This should
/// be quickly deprecated for something more formal.
struct BufferPathDelay {
  /// MILP variable denoting the buffer presence of a buffer type on a different
  /// signal.
  GRBVar &present;
  /// Combinational delay (in ns) introduced by the buffer, if present.
  double delay;

  /// Simple member-by-member constructor.
  BufferPathDelay(GRBVar &present, double delay = 0.0)
      : present(present), delay(delay){};
};

/// Holds the state and logic for FPL'22 smart buffer placement. This MILP
/// operates on the channels and units from a single CFDFC union derived from
/// the set of CFDFCs identified for a Handshake function. It takes into account
/// all timing domains of dataflow circuits (data, valid, ready) to derive an
/// optimal buffer placement. It creates
/// 1. custom channel constraints derived from channel-specific buffering
///    properties
/// 2. path constraints on all timing domains (including mixed-domain
///    connections within units)
/// 3. elasticity constraints
/// 4. throughput constraints for all CFDFCs that are part of the CFDFC union
/// 5. a maximixation objective, that rewards high CFDFC throughputs and
///    penalizes the placement of many large buffers in the circuit
class FPL22Buffers : public BufferPlacementMILP {
public:
  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. If a channel's buffering properties are provably
  /// unsatisfiable, the MILP will not be marked ready for optimization,
  /// ensuring that further calls to `optimize` fail.
  FPL22Buffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
               double targetPeriod, CFDFCUnion &cfUnion);

  /// Achieves the same as the other constructor but additionally logs placement
  /// decisions and achieved throughputs using the provided logger, and dumps
  /// the MILP model and solution at the provided name next to the log file.
  FPL22Buffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
               double targetPeriod, CFDFCUnion &cfUnion, Logger &logger,
               StringRef milpName);

protected:
  /// Interprets the MILP solution to derive buffer placement decisions. Since
  /// the MILP cannot encode the placement of both opaque and transparent slots
  /// on a single channel, some "interpretation" of the results is necessary to
  /// derive "mixed" placements where some buffer slots are opaque and some are
  /// transparent.
  void extractResult(BufferPlacement &placement) override;

private:
  /// The CFDFC union over which the MILP is described. Constraints are only
  /// created over the channels and units that are part of this union.
  CFDFCUnion &cfUnion;

  /// Adds channel-specific buffering constraints that were parsed from IR
  /// annotations to the Gurobi model.
  void addCustomChannelConstraints(Value channel);

  /// Adds path constraints for a specific signal of the provided channel.
  /// At the moment these *do not* take into account channel delays as may be
  /// specified in the channel's buffering properties.
  ///
  /// It is only valid to call this method after having added variables for the
  /// channel to the model.
  void addChannelPathConstraints(Value channel, SignalType type,
                                 const BufferPathDelay &otherBuffer);

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