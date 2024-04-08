//===- HandshakeDialect.cpp - Handshake dialect declaration -----*- C++ -*-===//
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
// This file implements the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace dynamatic;

void handshake::HandshakeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dynamatic/Dialect/Handshake/Handshake.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "dynamatic/Dialect/Handshake/HandshakeTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.cpp.inc"
      >();
}

// Provide implementations for the attributes, enums, and types that we use
#include "dynamatic/Dialect/Handshake/HandshakeDialect.cpp.inc"
#include "dynamatic/Dialect/Handshake/HandshakeEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "dynamatic/Dialect/Handshake/HandshakeTypes.cpp.inc"
