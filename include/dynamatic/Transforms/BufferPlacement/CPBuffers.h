//===- CPBuffers.h - Critical-path-only buffer placement --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CPBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CPBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferPlacementMILP.h"

namespace dynamatic {
namespace buffer {
namespace cpbuf {

/// Places the minimum number of buffers needed to satisfy the target clock
/// period. Unlike FPGA20, this formulation has no throughput objective.
class CPBuffers : public BufferPlacementMILP {
public:
  CPBuffers(CPSolver::SolverKind solverKind, int timeout, FuncInfo &funcInfo,
            const TimingDatabase &timingDB, double targetPeriod,
            StringRef writeTo = "");

protected:
  void extractResult(BufferPlacement &placement) override;

private:
  void addCustomChannelConstraints(Value channel);
  void setup();
};

} // namespace cpbuf
} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CPBUFFERS_H
