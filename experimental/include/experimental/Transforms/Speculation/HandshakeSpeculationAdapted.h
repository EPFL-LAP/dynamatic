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

#ifndef DYNAMATIC_TRANSFORMS_SPECULATION_ADAPTED_H
#define DYNAMATIC_TRANSFORMS_SPECULATION_ADAPTED_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {
namespace experimental {
namespace speculation {

#define GEN_PASS_DECL_HANDSHAKESPECULATIONADAPTED
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_ADAPTED_H
