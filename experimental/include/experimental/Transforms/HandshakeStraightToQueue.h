//===- HandshakeStraightToQueue.h - Implement S2Q algorithm --*- C++ -*----===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the pass which allows to implement straight to the queue,
// a different way of allocating basic blocks in the LSQ, based on an ASAP
// approach rather than relying on the network of cmerges.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKESTRAIGHTTOQUEUE_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKESTRAIGHTTOQUEUE_H
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

std::unique_ptr<dynamatic::DynamaticPass> createStraightToQueue();

#define GEN_PASS_DECL_HANDSHAKESTRAIGHTTOQUEUE
#define GEN_PASS_DEF_HANDSHAKESTRAIGHTTOQUEUE
#include "experimental/Transforms/Passes.h.inc"

} // namespace ftd
} // namespace experimental
} // namespace dynamatic
#endif // DYNAMATIC_TRANSFORMS_HANDSHAKESTRAIGHTTOQUEUE_H
