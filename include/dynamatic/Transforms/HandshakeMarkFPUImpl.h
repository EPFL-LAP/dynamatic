//===- HandshakeMarkFPUImpl.h - Replace memories ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-mark-fpu-impl pass
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEREMARKFPUIMPL_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEREMARKFPUIMPL_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {

// include tblgen base class declaration,
// options struct
// and pass create function
#define GEN_PASS_DECL_HANDSHAKEMARKFPUIMPL
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEREMARKFPUIMPL_H
