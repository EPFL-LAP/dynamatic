//===- HandshakeCanonicalize.h - Canonicalize Handshake ops -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-canonicalize pass.
//
// We use this pass instead of MLIR's generic canonicalization pass to have
// fine-grained control over the rewriter patterns we apply on the IR.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKECANONICALIZE_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKECANONICALIZE_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKECANONICALIZE
#define GEN_PASS_DEF_HANDSHAKECANONICALIZE
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeCanonicalize();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKECANONICALIZE_H