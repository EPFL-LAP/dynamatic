//===- HandshakeHoistExtInstances.h - Instances to IO -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-hoist-ext-instances pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEHOISTEXTINSTANCES
#define DYNAMATIC_TRANSFORMS_HANDSHAKEHOISTEXTINSTANCES

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEHOISTEXTINSTANCES
#define GEN_PASS_DEF_HANDSHAKEHOISTEXTINSTANCES
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeHoistExtInstances();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEHOISTEXTINSTANCES