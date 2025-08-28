//===- MaterializationUtil.cpp-Utility for materialized circuits-*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for materialized circuits.
//
//===----------------------------------------------------------------------===//

#include "MaterializationUtil.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

Value getForkTop(Value value) {
  if (auto fork = dyn_cast<ForkOp>(value.getDefiningOp())) {
    return getForkTop(fork.getOperand());
  }
  return value;
}

void iterateUsersOverNestedForkResults(Value result,
                                       llvm::SmallVector<Operation *> &users) {
  for (Operation *user : result.getUsers()) {
    if (auto forkOp = dyn_cast<ForkOp>(user)) {
      for (Value forkResult : forkOp.getResults()) {
        iterateUsersOverNestedForkResults(forkResult, users);
      }
    } else {
      users.push_back(user);
    }
  }
}

void iterateUsesOverNestedForkResults(Value result,
                                      llvm::SmallVector<OpOperand *> &uses) {
  for (OpOperand &use : result.getUses()) {
    if (auto forkOp = dyn_cast<ForkOp>(use.getOwner())) {
      for (Value forkResult : forkOp.getResults()) {
        iterateUsesOverNestedForkResults(forkResult, uses);
      }
    } else {
      uses.push_back(&use);
    }
  }
}

void eraseOpRecursively(Operation *op) {
  for (auto res : op->getResults()) {
    for (Operation *user : res.getUsers()) {
      eraseOpRecursively(user);
    }
  }
  op->erase();
}

void materializeValue(Value val) {
  Value forkTop = getForkTop(val);

  SmallVector<OpOperand *> uses;
  iterateUsesOverNestedForkResults(forkTop, uses);

  OpBuilder builder(forkTop.getContext());
  builder.setInsertionPointAfterValue(forkTop);

  if (uses.empty()) {
    SinkOp sinkOp = builder.create<SinkOp>(val.getLoc(), val);
    inheritBB(val.getDefiningOp(), sinkOp);
    return;
  }

  unsigned numUses = std::distance(uses.begin(), uses.end());

  if (numUses == 1)
    return;

  ForkOp forkOp = builder.create<ForkOp>(val.getLoc(), val, numUses);
  inheritBB(val.getDefiningOp(), forkOp);

  int i = 0;
  for (OpOperand *opOperand : llvm::make_early_inc_range(uses)) {
    if (opOperand->getOwner() == forkOp)
      continue;
    opOperand->set(forkOp.getResult()[i]);
    i++;
  }

  // Erase old forks
  for (Operation *user : val.getUsers()) {
    if (user == forkOp)
      continue;
    eraseOpRecursively(user);
  }
}

Operation *getUniqueUser(Value val) {
  assertMaterialization(val);
  return *val.getUsers().begin();
}

bool equalsIndirectly(Value a, Value b) {
  return getForkTop(a) == getForkTop(b);
}

void eraseMaterializedOperation(Operation *op) {
  for (OpResult result : op->getResults()) {
    if (result.use_empty())
      continue;
    Operation *user = getUniqueUser(result);
    if (auto sinkOp = dyn_cast<SinkOp>(user)) {
      sinkOp->erase();
    } else if (auto forkOp = dyn_cast<ForkOp>(user)) {
      eraseMaterializedOperation(forkOp);
    } else {
      op->emitError("Op has still uses, cannot be erased");
      llvm_unreachable("MaterializationUtil failed");
    }
  }
  op->erase();
}

void assertMaterialization(Value val) {
  if (val.hasOneUse())
    return;
  val.getDefiningOp()->emitError("Expected the value to be materialized, but "
                                 "it has zero or multiple users");
  llvm_unreachable("MaterializationUtil failed");
}

llvm::SmallVector<Operation *> iterateOverPossiblyIndirectUsers(Value result) {
  llvm::SmallVector<Operation *> users;
  iterateUsersOverNestedForkResults(getForkTop(result), users);
  return users;
}

Operation *getIndirectDefiningOp(Value value) {
  return getForkTop(value).getDefiningOp();
}
