//===- HandshakeOptimizeBitwidths.h - Optimize channel widths ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-optimize-bitwidths pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEOPTIMIZEBITWIDTHS_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEOPTIMIZEBITWIDTHS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEOPTIMIZEBITWIDTHS
#define GEN_PASS_DEF_HANDSHAKEOPTIMIZEBITWIDTHS
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeOptimizeBitwidths(bool legacy = false);

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEOPTIMIZEBITWIDTHS_H
