//===- HandshakeCanonicalize.cpp - Handshake canonicalization ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements helper functions for canonicalizing Handshake functions.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeCanonicalize.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace dynamatic;

bool dynamatic::hasRealUses(Value val) {
  return llvm::any_of(val.getUsers(), [&](Operation *user) {
    return !isa<handshake::SinkOp>(user);
  });
}

void dynamatic::eraseSinkUsers(Value val) {
  for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
    if (isa<handshake::SinkOp>(user))
      user->erase();
  }
}

void dynamatic::eraseSinkUsers(Value val, PatternRewriter &rewriter) {
  for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
    if (isa<handshake::SinkOp>(user))
      rewriter.eraseOp(user);
  }
}
