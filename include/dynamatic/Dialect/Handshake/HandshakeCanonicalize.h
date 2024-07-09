//===- HandshakeCanonicalize.h - Handshake canonicalization -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares helper functions for canonicalizing Handshake functions.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_CANONICALIZE_H
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_CANONICALIZE_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

/// Determines whether the given value has any "real" use i.e., a use which is
/// not the operand of a sink. If this function returns false for a given value
/// and one decides to erase the operation that defines it, one should keep in
/// mind that there may still be actual users of the value in the IR. In this
/// situation, using `eraseSinkUsers` in conjunction with this function will get
/// rid of all of the value's users.
bool hasRealUses(Value val);

/// Erases all sink operations that have the given value as operand.
void eraseSinkUsers(Value val);

/// Erases all sink operations that have the given value as operand. Uses the
/// rewriter to erase operations.
void eraseSinkUsers(Value val, PatternRewriter &rewriter);

} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_CANONICALIZE_H
