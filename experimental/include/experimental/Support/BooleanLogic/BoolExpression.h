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

/// Variable: x,y, z,...
/// AND: & or .
/// Or: | or +
/// Not: ! or ~
/// Remark: NOT perator is an INTEMEDIATE operator. That is, in all returned
/// Boolean Expressions, it's not present. The NOT operator is only used in the
/// parsig procedure and is then removed from the BoolExpression by the function
/// propagateNegation which propagates a NOT operator by applying DeMorgan's
/// Law.
enum class ExpressionType { Variable, Or, And, Not, Zero, One, End };

/// recursie function that replaces don't cares with 0 and 1
void replaceDontCaresRec(std::string s, std::set<std::string> &minterms);

/// wrapper function that replaces all don't cares in a set of minterms
std::set<std::string> replaceDontCares(const std::set<std::string> &minterms);

struct BoolExpression {
  ExpressionType type;

  BoolExpression(ExpressionType t) : type(t) {}

  virtual ~BoolExpression() = default;

  /// Prints BooolExpression tree in inorder traversal
  /// Inspired from
  /// https://www.geeksforgeeks.org/print-binary-tree-2-dimensions/
  void print(int space = 0);

  /// Converts a BoolExpression tree into its string representation
  std::string toString();

  /// Retrieve all the variables inside a BoolExpression
  std::set<std::string> getVariables();

  //---------Generating Truth Table based on Mintems (SOP Only)-----------

  /// Generates the minterms based n the variables
  /// 'd': don't care, 'n':null/void
  /// If a variable is a don't care: it should be set to 0 if it's negated and
  /// to 1 if it's non-negeated (for the minterm to be 1) If a variable is set
  /// to 0: if it's present in non-negated form in the minterm, then the minterm
  /// requires the variable to be both 0 and 1 for it to evaluate to 1, hence
  /// the minterm is void (no setting of the values gives a 1) Similarly if a
  /// variable is set to 1 and is present in negated form in the minterm
  void generateMintermVariable(std::string &s,
                               std::map<llvm::StringRef, int> varIndex);

  /// Generates the minterms of a specfic AND node
  void generateMintermAnd(std::string &s,
                          const std::map<llvm::StringRef, int> &varIndex);

  /// Function that generates the minterms of a BoolExpression
  void generateMintermsOr(unsigned numOfVariables,
                          const std::map<llvm::StringRef, int> &varIndex,
                          std::set<std::string> &minterms);

  /// Generates the truth table for a BoolExpression based on the
  /// minterms generated in generateMinterms
  std::set<std::string> generateTruthTable();

  //--------------Library Helper Functions--------------

  /// Propagates a not operator recursively in a BoolExpression tree
  /// by applying DeMorgan's law
  BoolExpression *propagateNegation(bool negated);

  /// Runs espresso logic minimzer on a BoolExpression
  std::string runEspresso();

  //--------------Library APIs--------------

  /// Converts String to BoolExpression
  /// parses an expression such as c1. c2.c1+~c3 . c1 + c4. ~c1 and stores it in
  /// the tree structure
  static BoolExpression *parseSop(llvm::StringRef strSop);

  /// Converts BoolExpression to String
  std::string sopToString();

  /// Creates an expression of a single variable
  /// Returns a dynamically-allocated variable
  static BoolExpression *boolVar(std::string id);

  /// AND two expressions
  /// Returns a dynamically-allocated variable
  static BoolExpression *boolAnd(BoolExpression *exp1, BoolExpression *exp2);

  /// OR two expressions
  /// Returns a dynamically-allocated variable
  static BoolExpression *boolOr(BoolExpression *exp1, BoolExpression *exp2);

  /// Negate an expression (apply DeMorgan's law)
  BoolExpression *boolNegate();

  /// Minimize an expression based on the espresso logic minimizer algorithm
  BoolExpression *boolMinimize();

private:
  // recursive helper for getVariables()
  void getVariablesRec(std::set<std::string> &s);
};

// This is specifically for operators: And, Or, Not
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

// This is specifically for signgle conditions: Variable, One, Zero
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
