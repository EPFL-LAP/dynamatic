//===- SpecAnnotatePaths.h - Annotate speculative paths ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --spec-annotate-paths pass. It is responsible for
// adding the attribute "speculative" to the operations in speculative paths.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SPEC_ANNOTATE_PATHS_H
#define DYNAMATIC_SPEC_ANNOTATE_PATHS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {
namespace speculation {

// Check if the given operand is annotated to be speculative
bool isSpeculative(mlir::OpOperand &operand, bool runAnalysis = false);

// Check if the given value is annotated to be speculative
bool isSpeculative(mlir::Value value, bool runAnalysis = false);

// Check if the given operation is annotated to be speculative
bool isSpeculative(mlir::Operation *op, bool runAnalysis = false);

std::unique_ptr<dynamatic::DynamaticPass> createSpecAnnotatePaths();

#define GEN_PASS_DECL_SPECANNOTATEPATHS
#define GEN_PASS_DEF_SPECANNOTATEPATHS
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_SPEC_ANNOTATE_PATHS_H
