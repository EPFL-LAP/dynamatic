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

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Transforms/SpeculationV2/HandshakeSpeculationV2.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

ForkOp flattenFork(ForkOp topFork);
void materializeValue(Value val);

/// Under the materialization, the user of the value should be unique (as long
/// as it's handshake-typed). This function returns the unique user of the
/// value.
Operation *getUniqueUser(Value val);

// Value forkValue(Value val);

bool equalsForContext(Value a, Value b);

void eraseMaterializedOperation(Operation *op);

void assertMaterialization(Value val);

llvm::SmallVector<Operation *>
iterateOverPossiblyMaterializedUsers(Value result);

Operation *getDefiningOpForContext(Value value);

#endif // DYNAMATIC_TRANSFORMS_MATERIALIZATION_UTIL_H
