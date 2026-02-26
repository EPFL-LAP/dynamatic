//===- HandshakeRemoveUnusedMemRef.h ----------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKERREMOVEUNUSEDMEMREF_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKERREMOVEUNUSEDMEMREF_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
#define GEN_PASS_DECL_HANDSHAKEREMOVEUNUSEDMEMREFS
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKERREMOVEUNUSEDMEMREF_H
