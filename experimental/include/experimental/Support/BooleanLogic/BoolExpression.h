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
// This also defines the interface for the library that handles boolean
// logic expressions.
// It includes functions for propagating negation through a tree using
// DeMorgan's law, and creating boolean expressions. Additionally, it provides
// functions to interact with the Espresso logic minimizer tool. The library
// supports basic operations on boolean expressions such as AND, OR, and
// negation.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_BOOLEXPRESSION_H
#define EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_BOOLEXPRESSION_H

#include "llvm/ADT/StringRef.h"
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
enum class ExpressionType { VARIABLE, OR, AND, NOT, ZERO, ONE, END };

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

  // function that gets all the variables inside a BoolExpression
  std::set<std::string> getVariables();

  //---------Generating Truth Table based on Mintems (SOP Only)-----------

  // Function that generates the minterms based n the variables
  void generateMintermVariable(std::string &s,
                               std::map<llvm::StringRef, int> varIndex);

  // Function that generates the minterms of a specfic AND node
  void generateMintermAnd(std::string &s,
                          const std::map<llvm::StringRef, int> &varIndex);

  // Function that generates the minterms of a BoolExpression
  void generateMintermsOr(unsigned numOfVariables,
                          const std::map<llvm::StringRef, int> &varIndex,
                          std::set<std::string> &minterms);

  // Fuction that generates the truth table for a BoolExpression based on the
  // minterms generated in generateMinterms
  std::set<std::string> generateTruthTable();

  //--------------Library Helper Functions--------------

  // Recursive function that propagates a not operator in a BoolExpression tree
  // by applying DeMorgan's law
  BoolExpression *propagateNegation(bool negated);

  // function to run espresso logic minimzer on a BoolExpression
  std::string runEspresso();

  //--------------Library APIs--------------

  // Convert String to BoolExpression
  // parses an expression such as c1. c2.c1+~c3 . c1 + c4. ~c1 and stores it in
  // the tree structure
  static BoolExpression *parseSop(llvm::StringRef strSop);

  // Convert BoolExpression to String
  std::string sopToString();

  // Create an expression of a single variable
  static BoolExpression *boolVar(std::string id);

  // AND two expressions
  static BoolExpression *boolAnd(BoolExpression *exp1, BoolExpression *exp2);

  // OR two expressions
  static BoolExpression *boolOr(BoolExpression *exp1, BoolExpression *exp2);

  // Negate an expression (apply DeMorgan's law)
  BoolExpression *boolNegate();

  // Minimize an expression based on the espresso logic minimizer algorithm
  BoolExpression *boolMinimize();

private:
  // recursive helper function for getVariables()
  void getVariablesRec(std::set<std::string> &s);
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
