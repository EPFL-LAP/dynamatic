//===- FPGA24Buffers.cpp ------------------------------------------------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
//
// TODO
//
//===--------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/FPGA24Buffers.h"

using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpl24;

FPGA24Buffers::FPGA24Buffers(CPSolver::SolverKind solverKind, int timeout,
                             FuncInfo &funcInfo, const TimingDatabase &timingDB,
                             double targetPeriod)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB,
                          targetPeriod) {
  /// TODO: Implement
}

void FPGA24Buffers::setup() {
  /// TODO: Implement
}