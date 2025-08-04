//===- HandshakeDialect.h - Handshake dialect declaration -------*- C++ -*-===//
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
// This file defines the Handshake MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_DIALECT_H
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_DIALECT_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

// Pull in the Dialect definition.
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h.inc"

// Pull in all types, attributes,
// and utility function declarations.
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h.inc"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_DIALECT_H
