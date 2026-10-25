//===- FuncMergeReturns.cpp - Give functions a single return ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --func-merge-returns pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_FUNCMERGERETURNS
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;

/// Gives the function a single return.
/// First we add a new basic block at the end of the cfg.
/// It takes the returned values from cf.br ops to its block args,
/// and then places a return op which reads from its block args
static void mergeReturns(func::FuncOp funcOp, OpBuilder &builder) {
  // get the original return ops
  SmallVector<func::ReturnOp> returnOps(funcOp.getOps<func::ReturnOp>());

  // need at least one-
  // even if the function returns no values,
  // the memory regions having completed all stores is an output of the kernel
  // Dynamatic needs an end op to explicitly encode this
  // and builds it from the return op
  assert(!returnOps.empty() && "no returns in function");

  // if there is only one return, no need to merge
  if (returnOps.size() == 1) {
    return;
  }

  // store the old insertion point
  OpBuilder::InsertionGuard guard(builder);
  // get the function body
  Region &body = funcOp.getBody();

  // handle MLIR "location per block-arg" stuff
  SmallVector<Location> argLocs(funcOp.getNumResults(), funcOp.getLoc());

  // make the extra last basic block with one block-arg
  // per result the function returns
  Block *exit = builder.createBlock(&body, body.end(),
                                    funcOp.getResultTypes(), argLocs);

  // add a return op fed by the block args
  builder.create<func::ReturnOp>(funcOp.getLoc(), exit->getArguments());

  // turn each of the original return ops into a cf::BranchOp
  for (func::ReturnOp returnOp : returnOps) {
    builder.setInsertionPoint(returnOp);
    builder.create<cf::BranchOp>(returnOp.getLoc(), exit,
                                 returnOp.getOperands());
    returnOp.erase();
  }
}

namespace {

/// Simple driver for the pass that gives every function a single return.
struct FuncMergeReturnsPass
    : public dynamatic::impl::FuncMergeReturnsBase<FuncMergeReturnsPass> {

  void runOnOperation() override {
    OpBuilder builder(&getContext());
    for (func::FuncOp funcOp : getOperation().getOps<func::FuncOp>()) {
      if (funcOp.isExternal())
        continue;
      mergeReturns(funcOp, builder);
    }
  };
};

} // namespace
