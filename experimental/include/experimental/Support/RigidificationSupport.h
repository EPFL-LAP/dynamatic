//===- RigidificationSupport.h - rigidification support ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declaration of the rigidification functionality that eliminates some control
// signals to reduce the handshake overhead
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"

mlir::LogicalResult rigidifyChannel(mlir::Value &channel,
                                    mlir::MLIRContext *ctx);