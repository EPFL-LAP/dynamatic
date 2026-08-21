//===- SystolicUnitReuseOptimization.h - Optimize systolic unit reuse -*- C++
//-*-===//
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

#ifndef DYNAMATIC_TRANSFORMS_SYSUNITREUSEOPTIMIZATION_H
#define DYNAMATIC_TRANSFORMS_SYSUNITREUSEOPTIMIZATION_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace dynamatic {

#define GEN_PASS_DECL_SYSTOLICUNITREUSEOPTIMIZATION
#define GEN_PASS_DEF_SYSTOLICUNITREUSEOPTIMIZATION
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createSystolicUnitReuseOptimization();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SYSUNITREUSEOPTIMIZATION_H
