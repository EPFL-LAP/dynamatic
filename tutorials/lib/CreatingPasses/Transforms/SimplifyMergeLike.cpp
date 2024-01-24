//===- SimplifyMergeLike.cpp - Simplifies merge-like operations -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --tutorial-handshake-simplify-merge-like pass, which uses a
// simple OpBuilder object to modify the IR within each handshake function.
//
//===----------------------------------------------------------------------===//

#include "tutorials/CreatingPasses/Transforms/SimplifyMergeLike.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace dynamatic;

/// Performs the simple transformation on the provided Handshake function,
/// deleting merges with a single input and downgrades control merges with an
/// unused index result into simpler merges.
static LogicalResult performSimplification(handshake::FuncOp funcOp,
                                           MLIRContext *ctx) {
  // Create an operation builder to allow us to create and insert new operation
  // inside the function
  OpBuilder builder(ctx);

  // Erase all merges with a single input
  for (handshake::MergeOp mergeOp :
       llvm::make_early_inc_range(funcOp.getOps<handshake::MergeOp>())) {
    if (mergeOp->getNumOperands() == 1) {
      // Replace all occurences of the merge's single result throughout the IR
      // with the merge's single operand. This is equivalent to bypassing the
      // merge
      mergeOp.getResult().replaceAllUsesWith(mergeOp.getOperand(0));
      // Erase the merge operation, whose result now has no uses
      mergeOp.erase();
    }
  }

  // Replace control merges with an unused index result into merges
  for (handshake::ControlMergeOp cmergeOp :
       llvm::make_early_inc_range(funcOp.getOps<handshake::ControlMergeOp>())) {

    // Get the control merge's index result (second result).
    // Equivalently, we could have written:
    //  auto indexResult = cmergeOp->getResult(1);
    // but using getIndex() is more readable and maintainable
    Value indexResult = cmergeOp.getIndex();

    // We can only perform the transformation if the control merge operation's
    // index result is not used throughout the IR
    if (!indexResult.use_empty())
      continue;

    // Now, we create a new merge operation at the same position in the IR as
    // the control merge we are replacing. The merge has the exact same inputs
    // as the control merge
    builder.setInsertionPoint(cmergeOp);
    handshake::MergeOp newMergeOp = builder.create<handshake::MergeOp>(
        cmergeOp.getLoc(), cmergeOp->getOperands());

    // Then, replace the control merge's first result (the selected input) with
    // the single result of the newly created merge operation
    Value mergeRes = newMergeOp.getResult();
    cmergeOp.getResult().replaceAllUsesWith(mergeRes);

    // Finally, we can delete the original control merge, whose results have
    // no uses anymore
    cmergeOp->erase();
  }

  return success();
}

namespace {

/// Simple pass driver for our merge-like simplification transformation. It will
/// apply the transformation on each function present in the matched MLIR
/// module.
struct SimplifyMergeLikePass
    : public dynamatic::tutorials::impl::SimplifyMergeLikeBase<
          SimplifyMergeLikePass> {

  void runOnOperation() override {
    // Get the MLIR context for the current operation being transformed
    MLIRContext *ctx = &getContext();
    // Get the operation being transformed (the top level module)
    ModuleOp mod = getOperation();

    // Iterate over all Handshake functions in the module
    for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>())
      // Perform the simple transformation individually on each function. In
      // case the transformation fails for at least a function, the pass should
      // be considered failed
      if (failed(performSimplification(funcOp, ctx)))
        return signalPassFailure();
  }
};
} // namespace

namespace dynamatic {
namespace tutorials {

/// Returns a unique pointer to an operation pass that matches MLIR modules. In
/// our case, this is simply an instance of our unparameterized
/// SimplifyMergeLikePass driver.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createSimplifyMergeLikePass() {
  return std::make_unique<SimplifyMergeLikePass>();
}
} // namespace tutorials
} // namespace dynamatic
