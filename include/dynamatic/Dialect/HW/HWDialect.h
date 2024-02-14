//===- HWDialect.h - HW dialect declaration ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file defines an HW MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HW_HW_DIALECT_H
#define DYNAMATIC_DIALECT_HW_HW_DIALECT_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"

// Pull in the dialect definition.
#include "dynamatic/Dialect/HW/HWDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "dynamatic/Dialect/HW/HWEnums.h.inc"

#endif // DYNAMATIC_DIALECT_HW_HW_DIALECT_H
