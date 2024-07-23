//===- Shannon.h - Shannon Decomposition for Boolean Expressions ----------*-
// C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures and functions for performing Shannon
// decomposition on boolean logic expressions. The Shannon decomposition is a
// method used to simplify boolean expressions by breaking them down into
// simpler parts.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_SHANNON_H
#define DYNAMATIC_SUPPORT_SHANNON_H

#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include <optional>

namespace dynamatic {
namespace experimental {
namespace boolean {

/// Forward declaration of the MUX struct.
struct MUX;

/// Represents either a boolean expression or a MUX.
struct Data {
  MUX *mux;
  std::optional<BoolExpression *> boolexpression;

  /// Constructor for initializing with a MUX.
  Data(MUX *m) : mux(m) {}

  /// Constructor for initializing with a boolean expression.
  Data(BoolExpression *b) : boolexpression(b) {}

  /// Prints the content of the Data based on its type.
  void print();
};

/// Represents a multiplexer for boolean logic expressions.
struct MUX {
  Data in0;  ///< Input 0 for the MUX.
  Data in1;  ///< Input 1 for the MUX.
  Data cond; ///< Condition for selecting the input.

  /// Constructor for initializing the MUX with inputs and condition.
  MUX(Data input0, Data input1, Data condition)
      : in0(input0), in1(input1), cond(condition) {}

  /// Prints the content of the MUX.
  void print();
};

/// Replaces a variable with a specified value in a boolean expression.
void replaceVarWithValue(BoolExpression *exp, const std::string &var,
                         ExpressionType t);

/// Performs Shannon expansion for the positive cofactor of a boolean
/// expression.
BoolExpression *shannonExpansionPositive(BoolExpression *exp,
                                         const std::string &var);

/// Performs Shannon expansion for the negative cofactor of a boolean
/// expression.
BoolExpression *shannonExpansionNegative(BoolExpression *exp,
                                         const std::string &var);

/// Applies Shannon decomposition on a boolean expression based on a list of
/// cofactors.
Data applyShannon(BoolExpression *exp,
                  const std::vector<std::string> &cofactorList);

} // namespace boolean
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_SHANNON_H
