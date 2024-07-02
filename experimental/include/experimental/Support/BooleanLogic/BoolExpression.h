//===- BoolExpression.h - Boolean expressions -------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the datastrucure BoolExpression to represent a boolean
// logical expresion as a tree.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_BOOLEXPRESSION_H
#define EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_BOOLEXPRESSION_H

#include <map>
#include <set>
#include <string>
#include <utility>

namespace dynamatic {
namespace experimental {
namespace boolean {

// Variable: x,y, z,...
// AND: & or .
// Or: | or +
// Not: ! or ~
// Remark: NOT perator is an INTEMEDIATE operator. That is, in all returned
// Boolean Expressions, it's not present. The NOT operator is only used in the
// parsig procedure and is then removed from the BoolExpression by the function
// propagateNegation which propagates a NOT operator by applying DeMorgan's Law.
enum class ExpressionType : int { Variable, Or, And, Not, Zero, One, End };

// recursie function that replaces don't cares with 0 and 1
void replaceDontCaresRec(std::string s, std::set<std::string> &minterms);

// wrapper function that replaces all don't cares in a set of minterms
std::set<std::string> replaceDontCares(const std::set<std::string> &minterms);

struct BoolExpression {
  ExpressionType type;

  BoolExpression(ExpressionType t) : type(t) {}

  virtual ~BoolExpression() = default;

  // Function to print BooolExpression tree in inorder traversal
  void print(int space = 0);

  // Function to convert a BoolExpression tree into its string representation
  std::string toString();

  // recursive helper function for getVariables()
  void getVariablesRec(std::set<std::string> &s);

  // function that gets all the variables inside a BoolExpression
  std::set<std::string> getVariables();

  //---------Generating Truth Table based on Mintems (SOP Only)-----------

  // Function that generates the minterms based n the variables
  void generateMintermVariable(std::string &s,
                               std::map<std::string, int> varIndex);

  // Function that generates the minterms of a specfic AND node
  void generateMintermAnd(std::string &s,
                          const std::map<std::string, int> &varIndex);

  // Function that generates the minterms of a BoolExpression
  void generateMintermsOr(int numOfVariables,
                          const std::map<std::string, int> &varIndex,
                          std::set<std::string> &minterms);

  // Fuction that generates the truth table for a BoolExpression based on the
  // minterms generated in generateMinterms
  std::set<std::string> generateTruthTable();
};

// This struct is specifically for operators: And, Or, Not
struct Operator : public BoolExpression {
  BoolExpression *left;
  BoolExpression *right;

  // Constructor for operator nodes
  Operator(ExpressionType t, BoolExpression *l, BoolExpression *r)
      : BoolExpression(t), left(l), right(r) {}

  ~Operator() {
    delete left;
    delete right;
  }
};

// This struct is specifically for signgle conditions: Variable, One, Zero
struct SingleCond : public BoolExpression {
  std::string id;
  bool isNegated;

  // Constructor for single condition nodes
  SingleCond(ExpressionType t, std::string i = "", bool negated = false)
      : BoolExpression(t), id(std::move(i)), isNegated(negated) {}
};

} // namespace boolean
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_BOOLEXPRESSION_H
