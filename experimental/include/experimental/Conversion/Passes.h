//===- Passes.h - Exp. conversion passes registration -----------*- C++ -*-===//
//
// This file contains declarations to register conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_CONVERSION_PASSES_H
#define EXPERIMENTAL_CONVERSION_PASSES_H

#include "experimental/Conversion/StandardToHandshakeFPL22.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace dynamatic {
namespace experimental {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "experimental/Conversion/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_CONVERSION_PASSES_H