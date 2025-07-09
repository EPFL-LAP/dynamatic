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

ForkOp flattenFork(ForkOp topFork) {
  SmallVector<Value> values;
  SmallVector<ForkOp> forksToBeErased;
  for (OpResult result : llvm::make_early_inc_range(topFork.getResults())) {
    materializeValue(result);
    Operation *user = getUniqueUser(result);
    if (auto forkOp = dyn_cast<ForkOp>(user)) {
      ForkOp newForkOp = flattenFork(forkOp);
      for (OpResult forkResult : newForkOp.getResults()) {
        assert(forkResult.hasOneUse());
        values.push_back(forkResult);
      }
      forksToBeErased.push_back(newForkOp);
    } else {
      values.push_back(result);
    }
  }

  if (values.size() == topFork.getResults().size())
    return topFork; // No need to flatten, already flat

  OpBuilder builder(topFork->getContext());
  builder.setInsertionPoint(topFork);
  ForkOp newForkOp = builder.create<ForkOp>(
      topFork.getLoc(), topFork.getOperand(), values.size());
  inheritBB(topFork, newForkOp);

  for (unsigned i = 0; i < values.size(); ++i) {
    values[i].replaceAllUsesWith(newForkOp.getResult()[i]);
  }

  for (ForkOp forkOp : forksToBeErased) {
    // Erase the old fork
    forkOp->erase();
  }
  topFork.erase();

  return newForkOp;
}

void materializeValue(Value val) {
  Operation *definingOp = val.getDefiningOp();
  OpBuilder builder(definingOp->getContext());
  builder.setInsertionPoint(definingOp);

  if (val.use_empty()) {
    SinkOp sinkOp = builder.create<SinkOp>(val.getLoc(), val);
    inheritBB(definingOp, sinkOp);
    return;
  }
  if (val.hasOneUse())
    return;

  unsigned numUses = std::distance(val.getUses().begin(), val.getUses().end());

  ForkOp forkOp = builder.create<ForkOp>(val.getLoc(), val, numUses);
  inheritBB(definingOp, forkOp);

  int i = 0;
  // To allow the mutation of operands, we use early increment range
  // TODO: Maybe he was not aware of this approach and the materialization pass
  // is dirty. Update it to use early increment range as well.
  for (OpOperand &opOperand : llvm::make_early_inc_range(val.getUses())) {
    if (opOperand.getOwner() == forkOp)
      continue;
    opOperand.set(forkOp.getResult()[i]);
    i++;
  }

  flattenFork(forkOp);
}

Operation *getUniqueUser(Value val) {
  if (!val.hasOneUse()) {
    val.getDefiningOp()->emitError("Expected the value to be materialized");
    for (Operation *user : val.getUsers()) {
      user->dump();
    }
    llvm_unreachable("MaterializationUtil failed");
  }

  return *val.getUsers().begin();
}

bool equalsForContext(Value a, Value b) {
  if (auto fork = dyn_cast<ForkOp>(a.getDefiningOp())) {
    return equalsForContext(fork.getOperand(), b);
  }
  if (auto fork = dyn_cast<ForkOp>(b.getDefiningOp())) {
    return equalsForContext(a, fork.getOperand());
  }
  return a == b;
}

void eraseMaterializedOperation(Operation *op) {
  for (OpResult result : op->getResults()) {
    if (result.use_empty())
      continue;
    Operation *user = getUniqueUser(result);
    if (auto forkOp = dyn_cast<ForkOp>(user)) {
      eraseMaterializedOperation(forkOp);
    } else {
      op->emitError("Op has still uses, cannot be erased");
      user->dump();
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

llvm::SmallVector<Operation *>
iterateOverPossiblyMaterializedUsers(Value result) {
  if (auto forkOp = dyn_cast<ForkOp>(result.getDefiningOp())) {
    // If the result is from a ForkOp, start iteration from the top of the fork.
    return iterateOverPossiblyMaterializedUsers(forkOp.getOperand());
  }

  llvm::SmallVector<Operation *> users;
  for (Operation *user : result.getUsers()) {
    if (auto forkUser = dyn_cast<ForkOp>(user)) {
      // If the user is a ForkOp, we need to iterate over its results
      for (Value forkResult : forkUser.getResults()) {
        for (Operation *forkUser : forkResult.getUsers()) {
          // Fork is not nested.
          if (isa<ForkOp>(forkUser)) {
            forkUser->emitError("Nested fork is not supported.");
            llvm_unreachable("MaterializationUtil failed");
          }
          users.push_back(forkUser);
        }
      }
    } else {
      users.push_back(user);
    }
  }
  return users;
}

Operation *getDefiningOpForContext(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (auto forkOp = dyn_cast<ForkOp>(definingOp)) {
    return getDefiningOpForContext(forkOp.getOperand());
  }
  return definingOp;
}
