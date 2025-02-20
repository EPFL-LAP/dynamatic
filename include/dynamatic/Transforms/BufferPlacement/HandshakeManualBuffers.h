//===- HandshakeManualBuffers.h - Manual buffers in DFG ---------*- C++ -*-===//
////
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Buffer placement in Handshake functions, using a set of available algorithms
// (all of which currently require Gurobi to solve MILPs). Buffers are placed to
// ensure circuit correctness and increase performance.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_HANDSHAKEMANUALBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_HANDSHAKEMANUALBUFFERS_H

#include "dynamatic/Support/DynamaticPass.h"
#include <memory>
namespace dynamatic {
namespace buffer {

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeSpeculation(const std::string &jsonPath = "");

#define GEN_PASS_DECL_HANDSHAKEPLACEBUFFERSPASS
#define GEN_PASS_DEF_HANDSHAKEPLACEBUFFERSPASS
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_HANDSHAKEMANUALBUFFERS_H
