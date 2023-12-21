//===- HandshakeMinimizeCstWidth.h - Min. constants bitwidth ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-minimize-cst-width pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZECSTWIDTH_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZECSTWIDTH_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEMINIMIZECSTWIDTH
#define GEN_PASS_DEF_HANDSHAKEMINIMIZECSTWIDTH
#include "dynamatic/Transforms/Passes.h.inc"

/// Computes the minimum required bitwidth needed to store the provided integer.
unsigned computeRequiredBitwidth(APInt val);

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeMinimizeCstWidth(bool optNegatives = false);

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZECSTWIDTH_H