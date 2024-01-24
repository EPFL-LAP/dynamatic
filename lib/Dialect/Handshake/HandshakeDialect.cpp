//===- HandshakeDialect.cpp - Handshake dialect declaration -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
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
// This file implements the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void HandshakeDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "dynamatic/Dialect/Handshake/Handshake.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#define GET_ATTRDEF_CLASSES
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.cpp.inc"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.cpp.inc"
#include "dynamatic/Dialect/Handshake/HandshakeEnums.cpp.inc"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
