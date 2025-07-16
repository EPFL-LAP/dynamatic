//===- CostAwareBuffers.h - Cost-aware buffer placement ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cost-aware smart buffer placement.
//
// This file declares the `CostAwareBuffers` class, which inherits the
// abstract `BufferPlacementMILP` class to setup and solve a real MILP from
// which buffering decisions can be made. Every public member declared in
// this file is under the `dynamatic::buffer::costaware` namespace, as to not
// create name conflicts for common structs with other implementors of
// `BufferPlacementMILP`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_COSTAWAREBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_COSTAWAREBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace dynamatic {
namespace buffer {
namespace costaware {

class CostAwareBuffers : public BufferPlacementMILP {
public:
  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. If a channel's buffering properties are provably
  /// unsatisfiable, the MILP will not be marked ready for optimization,
  /// ensuring that further calls to `optimize` fail.
  CostAwareBuffers(GRBEnv &env, FuncInfo &funcInfo,
                   const TimingDatabase &timingDB, double targetPeriod);

  CostAwareBuffers(GRBEnv &env, FuncInfo &funcInfo,
                   const TimingDatabase &timingDB, double targetPeriod,
                   Logger &logger, StringRef milpName);

protected:
  /// Interprets the MILP solution to derive buffer placement decisions.
  void extractResult(BufferPlacement &placement) override;

private:
  /// Adds channel-specific buffering constraints that were parsed from IR
  /// annotations to the Gurobi model.
  void addCustomChannelConstraints(Value channel);

  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

} // namespace costaware
} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_COSTAWAREBUFFERS_H