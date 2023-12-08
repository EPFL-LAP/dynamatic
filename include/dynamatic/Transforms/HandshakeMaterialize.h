//===- HandshakeMaterialize.h - Materialize Handshake IR --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-materialize pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEMATERIALIZE_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEMATERIALIZE_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEMATERIALIZE
#define GEN_PASS_DEF_HANDSHAKEMATERIALIZE
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeMaterialize();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEMATERIALIZE_H