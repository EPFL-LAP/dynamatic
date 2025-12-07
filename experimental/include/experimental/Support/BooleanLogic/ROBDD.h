//===- ROBDD.h - ROBDD construction and analysis ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a Reduced Ordered Binary Decision Diagram (ROBDD) builder
// and basic analysis utilities. A user-specified variable order is respected.
//
// Provided functionality:
//   * ROBDD Construction from a minimized BoolExpression and a variable order.
//   * Traversal of an ROBDD subgraph with a designated root and two designated
//     sinks.
//   * Enumeration of every vertex pair that covers all paths from the root to
//     either sink inside that subgraph.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_ROBDD_H
#define DYNAMATIC_SUPPORT_ROBDD_H

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
struct ROBDDNode {
  std::string var;
  unsigned falseSucc;
  unsigned trueSucc;
  std::vector<unsigned> preds;
};

/// Container for a reduced ordered BDD (ROBDD) built from a BoolExpression.
/// Each internal node is indexed by its position in the user-defined variable
/// order. Two additional nodes at the end are the 0 and 1 terminals.
class ROBDD {
public:
  /// Create an empty ROBDD.
  ROBDD();

  /// Build a lightweight ROBDD for CFG reachability analysis.
  ///
  /// Unlike a general ROBDD which may have multiple nodes for the same variable
  /// (representing different sub-functions), this specialized implementation
  /// enforces a **one-node-per-variable** constraint.
  ///
  /// This design exploits the specific structure of CFG path conditions,
  /// allowing for a simplified, linear-sized construction without the need for
  /// a global unique-table for node deduplication.
  mlir::LogicalResult
  buildROBDDFromExpression(BoolExpression *expr,
                           const std::vector<std::string> &varOrder);

  /// Traverse the subgraph reachable from `root` until either `tTrue` or
  /// `tFalse` (treated as local sinks). Aborts if any path reaches the global
  /// 0/1 terminals prematurely. Returns the list of node indices in the
  /// subgraph (sorted in ascending order, always including `root`, `tTrue`,
  /// `tFalse`).
  std::vector<unsigned> collectSubgraph(unsigned root, unsigned tTrue,
                                        unsigned tFalse) const;

  /// Enumerates all non-trivial pairs of vertices that cover all paths within
  /// the subgraph defined by (root, tTrue, tFalse). Trivial covers (e.g., those
  /// containing the root node or both terminals) are explicitly excluded from
  /// the results.
  ///
  /// Returns pairs sorted lexicographically (first ascending, then second
  /// ascending).
  std::vector<std::pair<unsigned, unsigned>>
  findPairsCoveringAllPaths(unsigned root, unsigned tTrue,
                            unsigned tFalse) const;

  /// Accessors.
  const std::vector<ROBDDNode> &getnodes() const { return nodes; }
  const std::vector<unsigned> &getpreds(unsigned idx) const {
    return nodes[idx].preds;
  }
  unsigned root() const { return rootIndex; }
  unsigned one() const { return oneIndex; }
  unsigned zero() const { return zeroIndex; }

private:
  /// Storage: internal nodes first, then terminals zeroIndex and oneIndex.
  std::vector<ROBDDNode> nodes;

  /// Variable order used to build the diagram (minimized & filtered).
  std::vector<std::string> order;

  /// Index of the ROBDD root node.
  unsigned rootIndex = 0;
  /// Index of the constant 0 terminal.
  unsigned zeroIndex = 0;
  /// Index of the constant 1 terminal.
  unsigned oneIndex = 0;

  /// Build an OBDD via Shannon expansion per the filtered order.
  void expandFrom(unsigned idx, BoolExpression *residual,
                  std::vector<char> &expanded);

  /// Helper: test whether nodes `a` and `b` cover all paths.
  bool doesPairCoverAllPaths(unsigned root, unsigned tTrue, unsigned tFalse,
                             unsigned a, unsigned b) const;
};

} // namespace boolean
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_ROBDD_H
