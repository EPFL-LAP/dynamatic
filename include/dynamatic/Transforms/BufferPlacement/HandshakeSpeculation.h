//===- HandshakeSpeculation.h - Speculation units placement -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-speculation pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_SPECULATION_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_SPECULATION_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace buffer {

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeSpeculation(
  StringRef srcOp = "", StringRef dstOp = "", bool dumpLogs = false);

#define GEN_PASS_DECL_HANDSHAKESPECULATION
#define GEN_PASS_DEF_HANDSHAKESPECULATION
#include "dynamatic/Transforms/Passes.h.inc"

/// Public pass driver for the speculation placement pass.
struct HandshakeSpeculationPass
    : public dynamatic::buffer::impl::HandshakeSpeculationBase<
          HandshakeSpeculationPass> {
  HandshakeSpeculationPass(StringRef srcOp, StringRef dstOp, bool dumpLogs);

  void runDynamaticPass() override;
};

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_SPECULATION_H
