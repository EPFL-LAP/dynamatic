//===- Passes.h - Exp. transformation passes registration -------*- C++ -*-===//
//
// This file contains the registration code for all experimental transformation
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_PASSES_H
#define EXPERIMENTAL_TRANSFORMS_PASSES_H

#include "dynamatic/Support/LLVM.h"
#include "experimental/Transforms/HandshakeFixArgNames.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_PASSES_H