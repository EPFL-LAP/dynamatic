//===- UnitMILPVars.h - Per-unit MILP variable storage ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Storage and accessors for the fluid-retiming MILP variables that belong to a
// single unit of a CFDFC, one per independent retiming path of the unit
// (see handshake::RetimingPathsOpInterface).
//
// The partition itself is provided by `buffer::getRetimingPaths`,
// which dispatches to the interface for ops that declare a partition and
// falls back to a single all-operands/all-results path otherwise.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_UTILS_UNITMILPVARS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_UTILS_UNITMILPVARS_H

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace dynamatic {
namespace buffer {

/// MILP variables and partition for one retiming path of a unit.
/// The occupancy of channels between units depends on
/// having at least one retiming variable for the path through the unit
/// that connects to that channel.  channels touching any
/// result in `results` tie to `retOut`. The unit throughput constraint
/// `throughput * latency == retOut - retIn` is added once per path.
struct RetPathVars {
  llvm::SmallSet<unsigned, 4> operands;
  llvm::SmallSet<unsigned, 4> results;
  std::optional<CPVar> retIn;
  std::optional<CPVar> retOut;
  std::optional<double> latency;

  /// Set the operands and results that belong to this timing path
  /// The retiming variables and latencies are not set in the constructor
  /// since you need access to the MILP model/timing model
  /// to get them, but they will be stored in here later
  explicit RetPathVars(RetimingPath &descriptor)
      : operands(descriptor.operands), results(descriptor.results) {}
};

/// Holds MILP variables associated to every unit in a CFDFC. Note that a unit
/// may appear in multiple CFDFCs and so may have multiple sets of these
/// variables. Every unit is modelled as one or more retiming paths:
/// Most units have one retiming path,
/// as all inputs are required to generate all outputs
///
/// Some special units (like the speculator) has multiple retiming paths
/// as there is more than 1 independent path tokens take through the unit
struct UnitVars {
  /// Pulls the paths through the unit from its tablegen
  /// but does not set any of the MILP variables
  explicit UnitVars(Operation *unit);

  /// The unit's retiming paths (at least one).
  llvm::SmallVector<RetPathVars> retPathVarList;

  /// Returns the fluid retiming variable of the path that owns the operand.
  CPVar &getRetIn(unsigned operandIdx);
  /// Returns the fluid retiming variable of the path that owns the result.
  CPVar &getRetOut(unsigned resultIdx);

  /// Returns the input-side fluid retiming variables
  llvm::SmallVector<CPVar> getInputRetimingVars();

  /// Returns the output-side fluid retiming variables
  llvm::SmallVector<CPVar> getOutputRetimingVars();

  /// Asserts that every retiming path has all additional variables
  /// (`retIn`, `retOut`, and `latency`) set.
  /// Call once before generating constraints that read these fields.
  void validate() const;

  /// [FPGA24] Occupancy contribution of this unit (real).
  CPVar occupancy;
};

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_UTILS_UNITMILPVARS_H
