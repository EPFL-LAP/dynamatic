//===- Backedge.cpp - Support for building backedges ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provide support for building backedges.
//
// This is taken directly from CIRCT, with minor modifications.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/Backedge.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace dynamatic;

BackedgeBuilder::BackedgeBuilder(OpBuilder &builder, Location loc)
    : builder(&builder), loc(loc) {}

BackedgeBuilder::BackedgeBuilder(PatternRewriter &rewriter, Location loc)
    : rewriter(&rewriter), loc(loc) {}

BackedgeBuilder::~BackedgeBuilder() { (void)clearOrEmitError(); }

LogicalResult BackedgeBuilder::clearOrEmitError() {
  unsigned numInUse = 0;
  for (Operation *op : edges) {
    if (!op->use_empty()) {
      mlir::InFlightDiagnostic diag = op->emitError("backedge of type ")
                                      << op->getResult(0).getType()
                                      << " is still in use";
      for (Operation *user : op->getUsers())
        diag.attachNote(user->getLoc()) << "used by " << *user;
      ++numInUse;
      continue;
    }
    if (rewriter)
      rewriter->eraseOp(op);
    else
      op->erase();
  }
  edges.clear();
  if (numInUse > 0)
    mlir::emitRemark(loc) << "Abandoned " << numInUse << " backedges";
  return success(numInUse == 0);
}

void BackedgeBuilder::abandon() { edges.clear(); }

Backedge BackedgeBuilder::get(Type resultType, mlir::LocationAttr optionalLoc) {
  if (!optionalLoc)
    optionalLoc = loc;
  // Create the opearion using either a builder or a rewriter
  Operation *op;
  if (rewriter)
    op = rewriter->create<mlir::UnrealizedConversionCastOp>(
        optionalLoc, resultType, ValueRange{});
  else
    op = builder->create<mlir::UnrealizedConversionCastOp>(
        optionalLoc, resultType, ValueRange{});
  edges.push_back(op);
  return Backedge(op);
}

Backedge::Backedge(mlir::Operation *op) : value(op->getResult(0)) {}

void Backedge::setValue(mlir::Value newValue) {
  assert(value.getType() == newValue.getType());
  assert(!set && "backedge already set to a value!");
  value.replaceAllUsesWith(newValue);
  set = true;
  value = newValue;
}
