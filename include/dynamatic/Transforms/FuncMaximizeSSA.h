//===- FuncMaximizeSSA.cpp - Maximal SSA form within functions --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --func-maximize-ssa pass.
//
// This if largely inherited from CIRCT, with minor modifications.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_FUNCMAXIMIZESSA_H
#define DYNAMATIC_TRANSFORMS_FUNCMAXIMIZESSA_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_FUNCMAXIMIZESSA
#define GEN_PASS_DEF_FUNCMAXIMIZESSA
#include "dynamatic/Transforms/Passes.h.inc"

/// Strategy class to control the behavior of SSA maximization. The class
/// exposes overridable filter functions to dynamically select which blocks,
/// block arguments, operations, and operation results should be put into
/// maximal SSA form. All filter functions should return true whenever the
/// entity they operate on should be considered for SSA maximization. By
/// default, all filter functions always return true.
class SSAMaximizationStrategy {
public:
  /// Determines whether a block should have the values it defines (i.e., block
  /// arguments and operation results within the block) SSA maximized.
  virtual bool maximizeBlock(Block &block);
  /// Determines whether a block argument should be SSA maximized.
  virtual bool maximizeArgument(BlockArgument arg);
  /// Determines whether an operation should have its results SSA maximized.
  virtual bool maximizeOp(Operation &op);
  /// Determines whether an operation's result should be SSA maximized.
  virtual bool maximizeResult(OpResult res);
  /// Default destructor.
  virtual ~SSAMaximizationStrategy() = default;
};

/// Determines whether the region is into maximal SSA form i.e., if all the
/// values within the region are in maximal SSA form.
bool isRegionSSAMaximized(Region &region);

/// Converts a single value within a function into maximal SSA form. This
/// removes any implicit dataflow of this specific value within the enclosing
/// function. The function adds new block arguments wherever necessary to carry
/// the value explicitly between blocks.
/// Succeeds when it was possible to convert the value into maximal SSA form.
LogicalResult maximizeSSA(Value value);

/// Considers all of an operation's results for SSA maximization, following a
/// provided strategy. This removes any implicit dataflow of the selected
/// operation's results within the enclosing function. The function adds new
/// block arguments wherever necessary to carry the results explicitly between
/// blocks. Succeeds when it was possible to convert the selected operation's
/// results into maximal SSA form.
LogicalResult maximizeSSA(Operation &op, SSAMaximizationStrategy &strategy);

/// Considers all values defined by a block (i.e., block arguments and operation
/// results within the block) for SSA maximization, following a provided
/// strategy. This removes any implicit dataflow of the selected values within
/// the enclosing function. The function adds new block arguments wherever
/// necessary to carry the values explicitly between blocks. Succeeds when it
/// was possible to convert the selected values defined by the block into
/// maximal SSA form.
LogicalResult maximizeSSA(Block &block, SSAMaximizationStrategy &strategy);

/// Considers all blocks within the region for SSA maximization, following a
/// provided strategy. This removes any implicit dataflow of the values defined
/// by selected blocks within the region. The function adds new block arguments
/// wherever necessary to carry the region's values explicitly between blocks.
/// Succeeds when it was possible to convert all of the values defined by
/// selected blocks into maximal SSA form.
LogicalResult maximizeSSA(Region &region, SSAMaximizationStrategy &strategy);

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFuncMaximizeSSA();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_FUNCMAXIMIZESSA_H