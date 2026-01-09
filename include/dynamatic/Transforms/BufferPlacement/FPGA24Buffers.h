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

#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"

namespace dynamatic {
namespace buffer {
namespace fpl24 {

class FPGA24Buffers : public BufferPlacementMILP {
protected:
  /// TODO: Explain
  FPGA24Buffers(CPSolver::SolverKind solverKind, int timeout,
                FuncInfo &funcInfo, const TimingDatabase &timingDB,
                double targetPeriod);

private:
  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

} // namespace fpl24
} // namespace buffer
} // namespace dynamatic

#endif /// DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPGA24BUFFERS_H