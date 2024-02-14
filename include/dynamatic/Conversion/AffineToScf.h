//===- AffineToScf.h - Convert Affine to SCF/standard dialect ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-affine-to-scf conversion pass, a slight
// variation of the --lower-affine pass from MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_AFFINETOSCF_H
#define DYNAMATIC_CONVERSION_AFFINETOSCF_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace dynamatic {

#define GEN_PASS_DECL_AFFINETOSCF
#define GEN_PASS_DEF_AFFINETOSCF
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createAffineToScfPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_AFFINETOSCF_H
