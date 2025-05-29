//===- Passes.h - Exp. analysis passes registration -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the registration code for all experimental analysis
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_PASSES_H
#define EXPERIMENTAL_ANALYSIS_PASSES_H

#include "dynamatic/Support/LLVM.h"
#include "experimental/Analysis/FormalPropertyAnnotation/HandshakeAnnotateProperties.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "experimental/Analysis/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_ANALYSIS_PASSES_H
