//===- HandshakeSetBufferingProperties.h - Set buf. props. ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-set-buffering-properties pass, which is
// useful to run before the buffer placement pass to make sure that all buffer
// placement constraints that can be deduced from the IR structure alone are
// part of the IR.
//
// It is possible, and certrainly necessary in some cases, to have other
// transformation passes tag the channels they care about with specific
// buffering properties as they go. However, there is a risk that these
// properties won't be fully honored in the intended way if further
// transformation passes alter tagged channel before buffer placement. While
// passes should do their best to preserve those properties when transforming
// the IR, some cases are likely to fall through or to be difficult to handle.
// This pass can serve as a "last transformation" before buffer placement and
// allow one to be sure that buffering properties set here are honored in the
// exact intended way.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_HANDSHAKESETBUFFERINGPROPERTIES_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_HANDSHAKESETBUFFERINGPROPERTIES_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/IR/DialectRegistry.h"

namespace dynamatic {
namespace buffer {

/// Updates the channel's buffering properties in the same way as it was done in
/// legacy Dynamatic's implementation of the initial smart buffer placement pass
/// (described in
/// https://www.epfl.ch/labs/lap/wp-content/uploads/2020/03/JosipovicFeb20_BuffePlacementAndSizingForHigh-PerformanceDataflowCircuits_FPGA20.pdf).
void setFPGA20Properties(circt::handshake::FuncOp funcOp);

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeSetBufferingProperties(const std::string &version = "fpga20");

#define GEN_PASS_DECL_HANDSHAKESETBUFFERINGPROPERTIES
#define GEN_PASS_DEF_HANDSHAKESETBUFFERINGPROPERTIES
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_HANDSHAKESETBUFFERINGPROPERTIES_H
