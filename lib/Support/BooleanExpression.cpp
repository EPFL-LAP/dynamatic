//===- BooleanExpression.cpp - Implementation of Boolean Expression Handling
//-----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the functionality for handling Boolean expressions
// represented as a tree structure. It includes methods to convert expressions
// to string format, generate truth tables, and print the expression tree in a
// structured format.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BooleanExpression.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cmath>
#include <set>
#include <string>

using namespace dynamatic;
using namespace llvm;

//----------BoolExpression Functions Implementatins--------

std::string BoolExpression::toString() {
  if (!this)
    return "";

  std::string s = "";

  if (type == ExpressionType::And || type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(this);
    s += op->left->toString();
  }

  if (type == ExpressionType::Variable) {
    SingleCond *singleCond = static_cast<SingleCond *>(this);
    if (singleCond->isNegated)
      s += "~";
    s += singleCond->id;
  } else if (type == ExpressionType::One) {
    s += "1";
  } else if (type == ExpressionType::Zero) {
    s += "0";
  } else if (type == ExpressionType::Or) {
    s += " + ";
  } else if (type == ExpressionType::And) {
    s += " . ";
  }

  if (type == ExpressionType::And || type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(this);
    s += op->right->toString();
  }

  return s;
}

void BoolExpression::getVariablesRec(std::set<std::string> &s) {
  if (!this) { // if null
    return;
  }
  if (type == ExpressionType::Variable) {
    SingleCond *cond = static_cast<SingleCond *>(this);
    s.insert(cond->id);
  } else if (type == ExpressionType::And || type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(this);
    op->left->getVariablesRec(s);
    op->right->getVariablesRec(s);
  }
}

std::set<std::string> BoolExpression::getVariables() {
  std::set<std::string> s;
  getVariablesRec(s);
  return s;
}

//---------Generating Truth Table based on Mintems (SOP Only)-----------

void dynamatic::replaceDontCaresRec(std::string s,
                                    std::set<std::string> &minterms) {
  int dontCareIndex = s.find('d');
  if (dontCareIndex == std::string::npos) {
    minterms.insert(s);
    return;
  }
  s[dontCareIndex] = '0';
  replaceDontCaresRec(s, minterms);
  s[dontCareIndex] = '1';
  replaceDontCaresRec(s, minterms);
}

std::set<std::string>
dynamatic::replaceDontCares(const std::set<std::string> &minterms) {
  std::set<std::string> mintermsWithoudDontCares;
  for (const std::string &s : minterms) {
    replaceDontCaresRec(s, mintermsWithoudDontCares);
  }
  return mintermsWithoudDontCares;
}

// 'd': don't care, 'n':null/void
// If a variable is a don't care: it should be set to 0 if it's negated and to 1
// if it's non-negeated (for the minterm to be 1) If a variable is set to 0: if
// it's present in non-negated form in the minterm, then the minterm requires
// the variable to be both 0 and 1 for it to evaluate to 1, hence the minterm is
// void (no setting of the values gives a 1) Similarly if a variable is set to 1
// and is present in negated form in the minterm
void BoolExpression::generateMintermVariable(
    std::string &s, std::map<std::string, int> varIndex) {
  SingleCond *singleCond = static_cast<SingleCond *>(this);
  if (s[varIndex[singleCond->id]] == 'd')
    s[varIndex[singleCond->id]] = singleCond->isNegated ? '0' : '1';
  else if (s[varIndex[singleCond->id]] == '0')
    s[varIndex[singleCond->id]] = singleCond->isNegated ? '0' : 'n';
  else if (s[varIndex[singleCond->id]] == '1')
    s[varIndex[singleCond->id]] = (!singleCond->isNegated) ? '1' : 'n';
}

void BoolExpression::generateMintermAnd(
    std::string &s, const std::map<std::string, int> &varIndex) {
  if (type == ExpressionType::Variable) {
    generateMintermVariable(s, varIndex);
  } else if (type == ExpressionType::Zero) { // 0 . exp =0 -> not a a minterm
    s[0] = 'n';
  } else if (type == ExpressionType::And || type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(this);
    if (op->left != nullptr)
      op->left->generateMintermAnd(s, varIndex);
    if (op->right != nullptr)
      op->right->generateMintermAnd(s, varIndex);
  }
}

void BoolExpression::generateMintermsOr(
    int numOfVariables, const std::map<std::string, int> &varIndex,
    std::set<std::string> &minterms) {
  if (!this) // if null
    return;
  if (type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(this);
    op->left->generateMintermsOr(numOfVariables, varIndex, minterms);
    op->right->generateMintermsOr(numOfVariables, varIndex, minterms);
  } else if (type == ExpressionType::One) { // 1 + exp = 1;
    std::string s(numOfVariables, 'd');
    minterms.insert(s);
  } else if (type == ExpressionType::Zero) { // 0 + exp = exp
    return;
  } else {
    std::string s(numOfVariables, 'd'); // initializing s
    generateMintermAnd(s, varIndex);
    if (s.find('n') ==
        std::string::npos) // no null in the minterm -> minterm is valid
      minterms.insert(s);
  }
}

// 1- generate all the minterms with don't cars
// 2- replace the don't cares wit 0s and 1s
// 3- loop over all rows in the truth table and check wether it corresponds to a
// minterm or not
std::set<std::string> BoolExpression::generateTruthTable() {
  std::set<std::string> variables = BoolExpression::getVariables();
  int numOfVariables = variables.size();
  std::map<std::string, int> varIndex;
  int index = 0;
  for (const std::string &s : variables) {
    varIndex[s] = index++;
  }
  // generate the minterms
  std::set<std::string> minterms;
  generateMintermsOr(numOfVariables, varIndex, minterms);
  std::set<std::string> mintermsWithoutDontCares = replaceDontCares(minterms);
  // generate the truth table
  std::string s(numOfVariables, 'd');
  std::set<std::string> r = {s};
  std::set<std::string> rows = replaceDontCares(r);
  std::set<std::string> table;
  for (const std::string &row : rows) {
    // if the row corresponds to the minterm, then f is 1
    if (mintermsWithoutDontCares.find(row) != mintermsWithoutDontCares.end())
      table.insert((row + "1"));
    // else f is 0
    else
      table.insert((row + "0"));
  }
  return table;
}

//--------Printing BoolExpresssion--------
static constexpr unsigned COUNT = 10;

// Inspired from https://www.geeksforgeeks.org/print-binary-tree-2-dimensions/
void BoolExpression::print(int space) {
  if (!this) // if null
    return;
  // Increase distance between levels
  space += COUNT;

  // Process right child first
  if (type == ExpressionType::Or || type == ExpressionType::And) {
    Operator *op = static_cast<Operator *>(this);
    op->right->print(space);
  }
  // Print current node after space
  // count
  llvm::outs() << "\n";
  for (int i = COUNT; i < space; i++)
    llvm::outs() << " ";

  if (type == ExpressionType::Variable) {
    SingleCond *singleCond = static_cast<SingleCond *>(this);
    if (singleCond->isNegated)
      llvm::outs() << "~";
    llvm::outs() << singleCond->id << "\n";
  } else if (type == ExpressionType::Or) {
    llvm::outs() << "+ " << "\n";
  } else if (type == ExpressionType::And) {
    llvm::outs() << ". " << "\n";
  } else if (type == ExpressionType::Zero) {
    llvm::outs() << "0 " << "\n";
  } else if (type == ExpressionType::One) {
    llvm::outs() << "1 " << "\n";
  }

  // Process left child
  if (type == ExpressionType::Or || type == ExpressionType::And) {
    Operator *op = static_cast<Operator *>(this);
    op->left->print(space);
  }
}
