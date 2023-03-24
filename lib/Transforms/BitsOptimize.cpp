//===- BitsOptimize.cpp -  Optimize bits width -------*- C++ -*-===//
//
// This file contains the implementation of the bits optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BitsOptimize.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"

using namespace mlir;
using namespace dynamatic;

static LogicalResult bitsOptimize(func::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  return success();
}

namespace {
struct HandshakeBitsOptimizePass
    : public HandshakeBitsOptimizeBase<HandshakeBitsOptimizePass> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    llvm::errs() << m << '\n';

  };
    // Process every function individually
    // for (auto funcOp : m.getOps<func::FuncOp>())
    //   if (failed(pushConstants(funcOp, &getContext())))
    //     return signalPassFailure();
  // };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createBitsOptimizationPass() {
  return std::make_unique<HandshakeBitsOptimizePass>();
}