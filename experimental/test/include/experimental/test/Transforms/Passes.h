//===- Passes.h - Experimental test passes registration ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the registration code for all experimental test passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TEST_TRANSFORMS_PASSES_H
#define EXPERIMENTAL_TEST_TRANSFORMS_PASSES_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {
namespace test {
/// Generate the code for registering passes.
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "experimental/test/Transforms/Passes.h.inc"
} // namespace test
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TEST_TRANSFORMS_PASSES_H
