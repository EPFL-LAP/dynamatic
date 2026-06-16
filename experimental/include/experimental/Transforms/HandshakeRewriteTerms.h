//===- HandshakeRewriteTerms.h - Rewrite Terms in Handshake Operation Sequences
//-----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-rewrite-terms pass.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_HANDSHAKEREWRITETERMS_H
#define EXPERIMENTAL_TRANSFORMS_HANDSHAKEREWRITETERMS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {

#define GEN_PASS_DECL_HANDSHAKEREWRITETERMS
#define GEN_PASS_DEF_HANDSHAKEREWRITETERMS
#include "experimental/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> rewriteHandshakeTerms();

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_HANDSHAKEREWRITETERMS_H
