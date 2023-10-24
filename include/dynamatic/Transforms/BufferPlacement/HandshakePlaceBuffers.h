//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace buffer {

/// Returns a string describing the meaning of the passed Gurobi optimization
/// status code. Descriptions are taken from
// https://www.gurobi.com/documentation/current/refman/optimization_status_codes.html
std::string getGurobiOptStatusDesc(int status);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(StringRef algorithm = "fpga20",
                                StringRef frequencies = "",
                                StringRef timingModels = "",
                                bool firstCFDFC = false, double targetCP = 4.0,
                                unsigned timeout = 180, bool dumpLogs = false);

#define GEN_PASS_DECL_HANDSHAKEPLACEBUFFERS
#define GEN_PASS_DEF_HANDSHAKEPLACEBUFFERS
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
