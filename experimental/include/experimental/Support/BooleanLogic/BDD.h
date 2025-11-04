//===- BDD.h - BDD construction and analysis --------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a Binary Decision Diagram (BDD) builder and
// basic analysis utilities. A user-specified variable order is respected.
//
// Provided functionality:
//   * BDD Construction from a minimized BoolExpression and a variable order.
//   * Traversal of a BDD subgraph with a designated root and two designated
//     sinks.
//   * Enumeration of every vertex pair that covers all paths from the root to
//     either sink inside that subgraph.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_BDD_H
#define DYNAMATIC_SUPPORT_BDD_H

#include <string>
#include <utility>
#include <vector>

#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Support/LogicalResult.h"

namespace dynamatic {
namespace experimental {
namespace boolean {

/// Replaces a boolean variable in a boolean expression with a truth value.
void restrict(BoolExpression *exp, const std::string &var,
              bool expressionValue);

/// One node of a reduced ordered BDD (ROBDD).
/// var       : variable name (empty for terminals)
/// falseSucc : successor index when var evaluates to false
/// trueSucc  : successor index when var evaluates to true
/// preds     : indices of all predecessor nodes that point to this node
struct BDDNode {
  std::string var;
  unsigned falseSucc;
  unsigned trueSucc;
  std::vector<unsigned> preds;
};

/// Container for a reduced ordered BDD built from a BoolExpression.
/// Each internal node is indexed by its position in the user-defined variable
/// order. Two additional nodes at the end are the 0 and 1 terminals.
class BDD {
public:
  /// Create an empty BDD.
  BDD();

  /// Build an ROBDD from a minimized boolean expression and a variable order.
  /// Variables in `varOrder` that do not appear in the expression are ignored.
  /// Returns `success()` on success, or `failure()` if the input is invalid.
  ///
  /// This is a simple method that assumes each variable appears only once in
  /// the BDD, which matches the case of checking whether a basic block in the
  /// CFG can be reached.
  mlir::LogicalResult
  buildROBDDFromExpression(BoolExpression *expr,
                           const std::vector<std::string> &varOrder);

  /// Traverse the subgraph reachable from `root` until either `tTrue` or
  /// `tFalse` (treated as local sinks). Aborts if any path reaches the global
  /// 0/1 terminals prematurely. Returns the list of node indices in the
  /// subgraph (sorted ascending, always including `root`, `tTrue`, `tFalse`).
  std::vector<unsigned> collectSubgraph(unsigned root, unsigned tTrue,
                                        unsigned tFalse) const;

  /// Enumerate all pairs of vertices that cover all paths within the subgraph
  /// defined by (root, tTrue, tFalse). Returns pairs sorted lexicographically
  /// (first ascending, then second ascending).
  std::vector<std::pair<unsigned, unsigned>>
  pairCoverAllPathsList(unsigned root, unsigned tTrue, unsigned tFalse) const;

  /// Accessors.
  const std::vector<BDDNode> &getnodes() const { return nodes; }
  const std::vector<unsigned> &getpreds(unsigned idx) const {
    return nodes[idx].preds;
  }
  unsigned root() const { return rootIndex; }
  unsigned one() const { return oneIndex; }
  unsigned zero() const { return zeroIndex; }

private:
  /// Storage: internal nodes first, then terminals zeroIndex and oneIndex.
  std::vector<BDDNode> nodes;

  /// Variable order used to build the diagram (minimized & filtered).
  std::vector<std::string> order;

  /// Index of the BDD root node.
  unsigned rootIndex = 0;
  /// Index of the constant 0 terminal.
  unsigned zeroIndex = 0;
  /// Index of the constant 1 terminal.
  unsigned oneIndex = 0;

  /// Build an OBDD via Shannon expansion per the filtered order.
  void expandFrom(unsigned idx, BoolExpression *residual,
                  std::vector<char> &expanded);

  /// Helper: test whether nodes `a` and `b` cover all paths.
  bool pairCoverAllPaths(unsigned root, unsigned tTrue, unsigned tFalse,
                         unsigned a, unsigned b) const;
};

} // namespace boolean
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_BDD_H
