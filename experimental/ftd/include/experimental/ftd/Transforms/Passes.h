//===- Passes.h - FTD transformation passes registration -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the registration code for all ftd transformation
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef FTD_TRANSFORMS_PASSES_H
#define FTD_TRANSFORMS_PASSES_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {
namespace ftd{

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "experimental/Transforms/Passes.h.inc"

}
} // namespace experimental
} // namespace dynamatic

#endif // FTD_TRANSFORMS_PASSES_H
