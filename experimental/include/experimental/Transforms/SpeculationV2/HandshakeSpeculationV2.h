//===- HandshakeSpeculationV2.h - Speculation units placement ---*- C++ -*-===//
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

#ifndef DYNAMATIC_TRANSFORMS_SPECULATION_PASS_V2_H
#define DYNAMATIC_TRANSFORMS_SPECULATION_PASS_V2_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <string>

namespace dynamatic {
namespace experimental {
namespace speculation {

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeSpeculationV2();

#define GEN_PASS_DECL_HANDSHAKESPECULATIONV2
#define GEN_PASS_DEF_HANDSHAKESPECULATIONV2
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_PASS_V2_H
