//===- ArithReduceStrength.h - Reduce strength of arith ops -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --arith-reduce-strength pass as well as some entities
// to model a tree of operations (e.g., a tree of adders to replace a multiplier
// with).
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_ARITHREDUCESTRENGTH_H
#define DYNAMATIC_TRANSFORMS_ARITHREDUCESTRENGTH_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include <variant>

namespace dynamatic {

struct OpTree;

/// Variant type representing an operand inside an operation tree (a node). It
/// is either:
/// - a std::shared_ptr<OpTree>: for the result of another operation tree
/// - an mlir::Value: for MLIR values
/// - a size_t: for constant values
using OpTreeOperand =
    std::variant<std::shared_ptr<OpTree>, mlir::Value, size_t>;

/// Shorthand for a vector of operation tree operands.
using TreeOperands = mlir::SmallVector<OpTreeOperand, 8>;

/// The "node type" of an operation tree, which is some kind of two-operands
/// operation.
enum OpType { ADD = 0, SUB = 1, SHIFT_LEFT = 2 };

/// Represents a binary tree of operations that ultimately results in a single
/// value being produced at the root. Each node of the tree represents some
/// two-operands operation (each of which may itself be the "result" of an
/// operation tree).
struct OpTree {
  /// The operation type that the root of this tree represents.
  const OpType opType;
  /// The left-hand-side operand to the operation.
  const OpTreeOperand left;
  /// The right-hand-side operand to the operation.
  const OpTreeOperand right;

  /// Maximum tree depth.
  const unsigned depth;
  /// Maximum adder depth (maximum number of sequential adders).
  const unsigned adderDepth;
  /// Number of nodes in the tree.
  const unsigned numNodes;

  /// Constructs an operation tree using moved l-value operands.
  OpTree(OpType opType, OpTreeOperand left, OpTreeOperand right);

  /// Builds the operation tree that replaces the provided operation in the IR.
  /// The tree is built at the location of the operation being replaced. Returns
  /// the single result of the root operation.
  Value buildTree(Operation *op, PatternRewriter &rewriter);

private:
  /// Returns the depth of a tree operand.
  unsigned getOperandDepth(OpTreeOperand &operand);

  /// Returns the adder depth of a tree operand.
  unsigned getOperandAdderDepth(OpTreeOperand &operand);

  /// Returns the number of nodes contained in a tree operand.
  unsigned getOperandNumNodes(OpTreeOperand &operand);

  /// Builds the operation tree recursively; first building each leaf, then
  /// their parent, then their parent's parent, etc., until reaching the root.
  /// At each step, caches intermediate results that may be reused in other
  /// parts of the tree (identical constants or identical sub-trees).
  Value buildTreeRecursive(
      Operation *op, PatternRewriter &rewriter,
      std::unordered_map<size_t, Value> &cstCache,
      std::unordered_map<std::shared_ptr<OpTree>, Value> &resultCache);
};

std::unique_ptr<dynamatic::DynamaticPass>
createArithReduceStrength(unsigned maxAdderDepthMul = 3);

#define GEN_PASS_DECL_ARITHREDUCESTRENGTH
#define GEN_PASS_DEF_ARITHREDUCESTRENGTH
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_ARITHREDUCESTRENGTH_H
