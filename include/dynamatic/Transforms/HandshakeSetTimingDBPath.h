//===- HandshakeSetTimingDBPath.h ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-set-timing-db-path pass
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKESETTIMINGDBPATH_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKESETTIMINGDBPATH_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {

#define GEN_PASS_DEF_HANDSHAKESETTIMINGDBPATH
#include "dynamatic/Transforms/Passes.h.inc"


} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKESETTIMINGDBPATH_H
