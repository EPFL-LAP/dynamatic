//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-place-buffers pass, including the pass's
// driver which may need to be inherited from by other passes that incorporate a
// buffer placement step within their logic.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_PLACEBUFFERS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace buffer {

std::unique_ptr<dynamatic::DynamaticPass> createHandshakePlaceBuffers(
    StringRef algorithm = "on-merges", StringRef frequencies = "",
    StringRef timingModels = "", bool firstCFDFC = false, double targetCP = 4.0,
    unsigned timeout = 180, bool dumpLogs = false);

#define GEN_PASS_DECL_HANDSHAKEPLACEBUFFERS
#define GEN_PASS_DEF_HANDSHAKEPLACEBUFFERS
#include "dynamatic/Transforms/Passes.h.inc"

/// Public pass driver for the buffer placement pass. Unlike most other
/// Dynamatic passes, users may wish to access the pass's internal state to
/// derive insights useful for different kinds of IR processing. To facilitate
/// users' workflow and minimize code duplication, this driver is public and
/// exposes most of its behavior in protected virtual methods which may be
/// overriden by sub-types of the pass.
struct HandshakePlaceBuffersPass
    : public dynamatic::buffer::impl::HandshakePlaceBuffersBase<
          HandshakePlaceBuffersPass> {

  /// Trivial field-by-field constructor.
  HandshakePlaceBuffersPass(StringRef algorithm, StringRef frequencies,
                            StringRef timingModels, bool firstCFDFC,
                            double targetCP, unsigned timeout, bool dumpLogs);

  /// Called on the MLIR module provided as input.
  void runDynamaticPass() override;

protected:
#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
  /// Called for all buffer placement strategies that not require Gurobi to
  /// be installed on the host system.
  LogicalResult placeUsingMILP();

  /// Checks a couple of invariants in the function that are required by our
  /// buffer placement algorithm. Fails when the function does not satisfy at
  /// least one invariant.
  virtual LogicalResult checkFuncInvariants(FuncInfo &info);

  /// Places buffers in the function, according to the logic dictated by the
  /// algorithm the pass was instantiated with.
  virtual LogicalResult placeBuffers(FuncInfo &info, TimingDatabase &timingDB);

  /// Identifies and extracts all existing CFDFCs in the function using
  /// estimated transition frequencies between its basic blocks. Fills the
  /// `cfdfcs` vector with the extracted cycles. CFDFC identification works by
  /// iteratively solving MILPs until the MILP solution indicates that no
  /// "executable cycle" remains in the circuit.
  virtual LogicalResult getCFDFCs(FuncInfo &info, Logger *logger,
                                  SmallVector<CFDFC> &cfdfcs);

  /// Computes an optimal buffer placement for a Handhsake function by solving
  /// a large MILP over the entire dataflow circuit represented by the
  /// function. Fills the `placement` map with placement decisions derived
  /// from the MILP's solution.
  virtual LogicalResult getBufferPlacement(FuncInfo &info,
                                           TimingDatabase &timingDB,
                                           Logger *logger,
                                           BufferPlacement &placement);
#endif
  /// Called for all buffer placement strategies that do not require Gurobi to
  /// be installed on the host system.
  LogicalResult placeWithoutUsingMILP();

  /// Instantiates buffers inside the IR, following placement decisions
  /// determined by the buffer placement MILP.
  virtual void instantiateBuffers(BufferPlacement &placement);
};

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_PLACEBUFFERS_H
