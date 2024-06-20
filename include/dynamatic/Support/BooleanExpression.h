//===- BooleanExpression.h - Definition for boolean expressions -----*- C++
//-*-===//
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

#ifndef DYNAMATIC_SUPPORT_BOOLEANEXPRESSION_H
#define DYNAMATIC_SUPPORT_BOOLEANEXPRESSION_H

#include <map>
#include <set>
#include <string>
#include <utility>

namespace dynamatic {

enum class ExpressionType : int { Variable, Or, And, Not, Zero, One, End };

// recursie function that replaces don't cares with 0 and 1
void replaceDontCaresRec(std::string s, std::set<std::string> &minterms);

// wrapper function that replaces all don't cares in a set of minterms
std::set<std::string> replaceDontCares(const std::set<std::string> &minterms);

struct BoolExpression {
  ExpressionType type;

  struct {
    std::string id;
    bool isNegated;
  } singleCond;

  BoolExpression *left;
  BoolExpression *right;

  BoolExpression(ExpressionType t)
      : type(t), singleCond{"", false}, left(nullptr), right(nullptr) {}

  // Constructor for single condition nodes
  BoolExpression(ExpressionType t, std::string i, bool negated = false)
      : type(t), singleCond{std::move(i), negated}, left(nullptr),
        right(nullptr) {}

  // Constructor for operator nodes
  BoolExpression(ExpressionType t, BoolExpression *l, BoolExpression *r)
      : type(t), singleCond{"", false}, left(l), right(r) {}

  ~BoolExpression() {
    delete left;
    delete right;
  }

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
  void generateMinterms(int numOfVariables,
                        const std::map<std::string, int> &varIndex,
                        std::set<std::string> &minterms);

  // Fuction that generates the truth table for a BoolExpression based on the
  // minterms generated in generateMinterms
  std::set<std::string> generateTruthTable();
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_BOOLEANEXPRESSION_H
