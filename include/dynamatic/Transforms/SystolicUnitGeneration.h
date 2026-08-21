//===- SystolicUnitGenerator.h - Generate systolic units --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --push-constants pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SYSUNITGEN_H
#define DYNAMATIC_TRANSFORMS_SYSUNITGEN_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace dynamatic {

#define GEN_PASS_DECL_SYSTOLICUNITGENERATION
#define GEN_PASS_DEF_SYSTOLICUNITGENERATION
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createSystolicUnitGeneration();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SYSUNITGEN_H
