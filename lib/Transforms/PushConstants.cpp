//===- PushConstants.cpp - Push constants in using blocks -------*- C++ -*-===//
//
// This file contains the implementation of the constant pushing pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/PushConstants.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"

using namespace mlir;
using namespace dynamatic;

/// Pushes all of a function's constants in using blocks.
static LogicalResult pushConstants(func::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);

  for (auto constantOp :
       llvm::make_early_inc_range(funcOp.getOps<arith::ConstantOp>())) {
    Block *defBlock = constantOp->getBlock();
    bool usedByDefiningBlock = false;

    // Determine blocks where the constant is used
    DenseMap<Block *, SmallVector<Operation *, 4>> usingBlocks;
    for (auto *user : constantOp.getResult().getUsers())
      if (auto block = user->getBlock(); block != defBlock)
        usingBlocks[block].push_back(user);
      else
        usedByDefiningBlock = true;

    // Create a new constant operation in every block where the constant is used
    for (auto &[block, users] : usingBlocks) {
      builder.setInsertionPointToStart(block);
      auto newCstOp = builder.create<arith::ConstantOp>(constantOp->getLoc(),
                                                        constantOp.getValue());
      for (auto user : users)
        user->replaceUsesOfWith(constantOp.getResult(), newCstOp.getResult());
    }

    // Delete the original constant operation if it isn't used
    if (!usedByDefiningBlock)
      constantOp->erase();
  }

  return success();
}

namespace {
struct ArithPushConstants : public ArithPushConstantsBase<ArithPushConstants> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    // Process every function individually
    for (auto funcOp : m.getOps<func::FuncOp>())
      if (failed(pushConstants(funcOp, &getContext())))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createArithPushConstants() {
  return std::make_unique<ArithPushConstants>();
}
