//===- HandshakePrepareForLegacy.h - Prepare for legacy flow ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-prepare-for-legacy pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEPREPAREFORLEGACY_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEPREPAREFORLEGACY_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<dynamatic::DynamaticPass> createHandshakePrepareForLegacy();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEPREPAREFORLEGACY_H
