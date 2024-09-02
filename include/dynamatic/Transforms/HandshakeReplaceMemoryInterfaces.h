//===- HandshakeReplaceMemoryInterfaces.h - Replace memories ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-replace-memory-interfaces pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEREPLACEMEMORYINTERFACES_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEREPLACEMEMORYINTERFACES_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEREPLACEMEMORYINTERFACES
#define GEN_PASS_DEF_HANDSHAKEREPLACEMEMORYINTERFACES
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeReplaceMemoryInterfaces();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEREPLACEMEMORYINTERFACES_H
