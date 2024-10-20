//===- Shannon.cpp - Shannon Decomposition for Boolean Expressions -*- C++ -*-//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the functions for performing Shannon decomposition on
// boolean logic expressions. It includes methods for replacing variables with
// values, performing positive and negative Shannon expansions, and applying
// Shannon decomposition.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/BooleanLogic/Shannon.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <string>
#include <vector>

using namespace dynamatic::experimental::boolean;
using namespace llvm;

void MultiplexerIn::print() {
  // Print according to the type of the multiplexer input
  if (boolexpression.has_value())
    boolexpression.value()->print();
  else if (mux)
    mux->print();
}

void Multiplexer::print() {
  llvm::dbgs() << "in0\n";
  in0->print();
  llvm::dbgs() << "in1\n";
  in1->print();
  llvm::dbgs() << "cond\n";
  cond->print();
}

/// Replaces a variable with a specified value in a boolean expression.
static void replaceVarWithValue(BoolExpression *exp, const std::string &var,
                                ExpressionType t) {

  // If the input is a variable, then you just need to substitute the same
  // variable with the provided value `t`. Otherwise, modify the left and
  // right side of the binary expression.
  if (exp->type == ExpressionType::Variable) {
    SingleCond *singleCond = static_cast<SingleCond *>(exp);
    if (singleCond->id == var) {
      if (singleCond->isNegated) {
        if (t == ExpressionType::One)
          exp->type = ExpressionType::Zero;
        if (t == ExpressionType::Zero)
          exp->type = ExpressionType::One;
      } else {
        exp->type = t;
      }
    }
  } else if (exp->type == ExpressionType::And ||
             exp->type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(exp);
    if (op->left)
      replaceVarWithValue(op->left, var, t);

    if (op->right)
      replaceVarWithValue(op->right, var, t);
  }
}

/// Performs Shannon expansion for the positive cofactor of a boolean
/// expression. This is done by replacing the vairable with 1 and minimizing the
/// expression.
static BoolExpression *shannonExpansionPositive(BoolExpression *exp,
                                                const std::string &var) {
  // First replace the value, then minimize
  replaceVarWithValue(exp, var, ExpressionType::One);
  return exp->boolMinimize();
}

/// Performs Shannon expansion for the negative cofactor of a boolean
/// expression. This is done by replacing the vairable with 0 and minimizing the
/// expression.
static BoolExpression *shannonExpansionNegative(BoolExpression *exp,
                                                const std::string &var) {
  // First replace the value, then minimize
  replaceVarWithValue(exp, var, ExpressionType::Zero);
  return exp->boolMinimize();
}

MultiplexerIn *dynamatic::experimental::boolean::applyShannon(
    BoolExpression *exp, const std::vector<std::string> &cofactorList) {
  // If the type of the expression is not a binary one, just return that value
  if (exp->type == ExpressionType::Variable ||
      exp->type == ExpressionType::Zero || exp->type == ExpressionType::One)
    return new MultiplexerIn(exp);

  // Start with the current element in the cofactor list
  const std::string &var = cofactorList[0];

  // Substitute `var` with zero
  BoolExpression *expCopyForNeg = exp->deepCopy();
  BoolExpression *neg = shannonExpansionNegative(expCopyForNeg, var);

  // Substitute `var` with one
  BoolExpression *expCopyForPos = exp->deepCopy();
  BoolExpression *pos = shannonExpansionPositive(expCopyForPos, var);

  // Get the remaning list of cofactors
  std::vector<std::string> subList(std::next(cofactorList.begin()),
                                   cofactorList.end());

  // Further apply shannon over the two sides of the expression, and use `var`
  // as select input signal of the multiplexer
  MultiplexerIn *in0 = applyShannon(neg, subList);
  MultiplexerIn *in1 = applyShannon(pos, subList);
  MultiplexerIn *select =
      new MultiplexerIn(new SingleCond(ExpressionType::Variable, var));

  Multiplexer *mux = new Multiplexer(in0, in1, select);

  return new MultiplexerIn(mux);
}
