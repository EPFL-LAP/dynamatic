//===- Passes.h - Exp. transformation passes registration -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the registration code for all experimental transformation
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_PASSES_H
#define EXPERIMENTAL_TRANSFORMS_PASSES_H

#include "dynamatic/Support/LLVM.h"
#include "experimental/Transforms/HandshakeFixArgNames.h"
#include "experimental/Transforms/Speculation/HandshakeSpeculation.h"
#include "experimental/Transforms/HandshakePlaceBuffersCustom.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_PASSES_H
