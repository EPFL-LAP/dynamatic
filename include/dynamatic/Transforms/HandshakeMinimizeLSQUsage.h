//===- HandshakeMiminizeLSQUsage.h - LSQ flow analysis ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-minimize-lsq-usage pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZELSQUSAGE_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZELSQUSAGE_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEMIMINIZELSQUSAGE
#define GEN_PASS_DEF_HANDSHAKEMIMINIZELSQUSAGE
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeMinimizeLSQUsage();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZELSQUSAGE_H