//===- MaterializationUtil.h - Utility for materialized circuits-*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for materialized circuits.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_MATERIALIZATION_UTIL_H
#define DYNAMATIC_TRANSFORMS_MATERIALIZATION_UTIL_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

/// Flatten nested forks. Returns the new top fork.
ForkOp flattenFork(ForkOp topFork);

/// Materialize the value (i.e., ensuring it has a single user), by introducing
/// a fork unit. Nested forks are flattened internally.
void materializeValue(Value val);

/// Under the materialization, the user of the value should be unique (as long
/// as it's handshake-typed). This function returns the unique user of the
/// value.
Operation *getUniqueUser(Value val);

/// Computes the equality while ignoring differences in fork results.
bool equalsIndirectly(Value a, Value b);

/// Erase the operation as well as downstream unused forks.
void eraseMaterializedOperation(Operation *op);

/// Asserts the value is materialized.
void assertMaterialization(Value val);

/// Returns the vector of indirect users of the value, including those that use
/// forked values.
llvm::SmallVector<Operation *> iterateOverPossiblyIndirectUsers(Value result);

/// Retrieves the top of fork tree.
Value getForkTop(Value value);

/// Retrieves the defining operation, ignoring any forks present between uses.
Operation *getIndirectDefiningOp(Value value);

#endif // DYNAMATIC_TRANSFORMS_MATERIALIZATION_UTIL_H
