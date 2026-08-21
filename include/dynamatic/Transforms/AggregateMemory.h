//===- AggregateMemory.h - Aggregate Memory Pass ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --aggregate-memory pass
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_AGGREGATEMEMORY_H
#define DYNAMATIC_TRANSFORMS_AGGREGATEMEMORY_H

#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace dynamatic {

#define GEN_PASS_DECL_AGGREGATEMEMORY
#define GEN_PASS_DEF_AGGREGATEMEMORY
#include "dynamatic/Transforms/Passes.h.inc"
std::unique_ptr<dynamatic::DynamaticPass> createAggregateMemoryPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_AGGREGATEMEMORY_H
