//===--------- ReadOnceBDD.h - Read-once BDD --------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a lightweight **Read-Once Binary Decision Diagram (BDD)**
// builder and basic analysis utilities. It constructs a decision diagram for a
// minimized boolean expression following a user-provided variable order. Each
// internal node corresponds to one boolean variable with two outgoing edges
// (false/true). Two terminal nodes (0 and 1) are always appended at the end.
//
// The class provides:
//   * Construction from a minimized BoolExpression.
//   * Traversal of a subgraph defined by a root and two designated sinks.
//   * Enumeration of all 2-vertex cuts in that subgraph.
//
// This implementation assumes the expression is already minimized and
// read-once compatible (no variable reused along any path).
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_READONCEBDD_H
#define EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_READONCEBDD_H

#include <string>
#include <utility>
#include <vector>

#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Support/LogicalResult.h"

namespace dynamatic {
namespace experimental {
namespace boolean {

/// One node of a read-once BDD.
/// - `var`       : variable name (empty string for terminal nodes)
/// - `falseSucc` : index of the successor if the variable evaluates to false
/// - `trueSucc`  : index of the successor if the variable evaluates to true
/// - `preds`     : indices of all predecessor nodes that point to this node
struct BDDNode {
  std::string var;
  unsigned falseSucc;
  unsigned trueSucc;
  std::vector<unsigned> preds;
};

/// Container for a Read-Once BDD.
/// Each internal node is indexed by its position in the user-defined variable
/// order. Two additional nodes at the end are the 0 and 1 terminals.
class ReadOnceBDD {
public:
  /// Create an empty BDD.
  ReadOnceBDD();

  /// Build the BDD from a minimized boolean expression and a variable order.
  /// Variables in `varOrder` that do not appear in the expression are ignored.
  /// Returns `success()` on success, or `failure()` if the input is invalid.
  mlir::LogicalResult
  buildFromExpression(BoolExpression *expr,
                      const std::vector<std::string> &varOrder);

  /// Traverse the subgraph reachable from `root` until either `tTrue` or
  /// `tFalse` (treated as local sinks). Aborts if any path reaches the global
  /// 0/1 terminals prematurely. Returns the list of node indices in the
  /// subgraph (sorted ascending, always including `root`, `tTrue`, `tFalse`).
  std::vector<unsigned> collectSubgraph(unsigned root, unsigned tTrue,
                                        unsigned tFalse) const;

  /// Enumerate all 2-vertex cuts within the subgraph defined by
  /// (root, tTrue, tFalse). Returns pairs sorted lexicographically
  /// (first ascending, then second ascending).
  std::vector<std::pair<unsigned, unsigned>>
  listTwoVertexCuts(unsigned root, unsigned tTrue, unsigned tFalse) const;

  /// Accessors
  const std::vector<BDDNode> &getnodes() const { return nodes; }
  const std::vector<unsigned> &getpreds(unsigned idx) const {
    return nodes[idx].preds;
  }
  unsigned root() const { return rootIndex; }
  unsigned one() const { return oneIndex; }
  unsigned zero() const { return zeroIndex; }

private:
  /// All nodes of the BDD: internal nodes first (following `order`),
  /// then the 0 terminal, then the 1 terminal.
  std::vector<BDDNode> nodes;

  /// Variable order actually used to build the diagram (minimized & filtered).
  std::vector<std::string> order;

  /// Index of the BDD root node.
  unsigned rootIndex = 0;
  /// Index of the constant 0 terminal.
  unsigned zeroIndex = 0;
  /// Index of the constant 1 terminal.
  unsigned oneIndex = 0;

  /// Recursively expand edges for a node according to Shannon expansion.
  /// Avoids duplicating nodes by reusing existing indices.
  void expandFrom(unsigned idx, BoolExpression *residual,
                  std::vector<char> &expanded);

  /// Helper: test whether banning nodes `a` and `b` disconnects both sinks.
  bool sinksUnreachableIfBan(unsigned root, unsigned tTrue, unsigned tFalse,
                             unsigned a, unsigned b) const;
};

} // namespace boolean
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_READONCEBDD_H