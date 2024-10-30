//===- BDD.h - BDD Decomposition for Bool Expressions --*-- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the functions for performing a BDD decomposition on
// boolean logic expression, following a specific order of variables.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_BDD_H
#define DYNAMATIC_SUPPORT_BDD_H

#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include <optional>

namespace dynamatic {
namespace experimental {
namespace boolean {

/// A BDD is a tree which represents a boolean expression by considering all the
/// outcomes of the expression as a decision tree, with two edges (positive and
/// negative) for each non terminal node. Each node is charaterized by a boolean
/// variable and possibly 2 subsequent nodes, stored into `inputs`
struct BDD {
  // First element is the `negative` branch; second is the `positive` branch
  std::optional<std::pair<BDD *, BDD *>> inputs;
  BoolExpression *boolVariable;

  /// Build a BDD node with two descendents
  BDD(BDD *ni, BDD *pi, BoolExpression *bv)
      : inputs({ni, pi}), boolVariable(bv) {}

  /// Build a leaf
  BDD(BoolExpression *bv) : boolVariable(bv) {}
};

/// Replaces a boolean variable in a boolean expression with the specified truth
/// value
void restrict(BoolExpression *exp, const std::string &var,
              bool expressionValue);

/// Build a BDD out of an expression, by relying on the order of variables
/// provided
BDD *buildBDD(BoolExpression *exp, std::vector<std::string> variableList);

} // namespace boolean
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_BDD_H
