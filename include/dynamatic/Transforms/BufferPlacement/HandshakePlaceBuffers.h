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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(const std::string &frequencies = "",
                                const std::string &timingModels = "",
                                bool firstCFDFC = false, double targetCP = 4.0,
                                unsigned timeout = 180, bool dumpLogs = false);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
