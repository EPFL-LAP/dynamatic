//===- BoolExpression.cpp - Boolean expressions -----------------*- C++ -*-===//
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
// It also includes propagating negation using DeMorgan's law, and
// creating basic boolean expressions. The implemented functions support basic
// boolean operations such as AND, OR, and negation, and enable minimization of
// boolean expressions using the Espresso algorithm.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/BooleanLogic/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <string>

using namespace dynamatic::experimental::boolean;
using namespace llvm;

//----------BoolExpression Functions Implementatins--------

std::string BoolExpression::toString() {
  std::string s;

  if (type == ExpressionType::AND || type == ExpressionType::OR) {
    Operator *op = static_cast<Operator *>(this);
    if (op->left)
      s += op->left->toString();
  }

  switch (type) {
  case ExpressionType::VARIABLE: {
    SingleCond *singleCond = static_cast<SingleCond *>(this);
    if (singleCond->isNegated)
      s += "~";
    s += singleCond->id;
    break;
  }
  case ExpressionType::ONE:
    s += "1";
    break;
  case ExpressionType::ZERO:
    s += "0";
    break;
  case ExpressionType::OR:
    s += " + ";
    break;
  case ExpressionType::AND:
    s += " . ";
    break;
  default:
    break;
  }

  if (type == ExpressionType::AND || type == ExpressionType::OR) {
    Operator *op = static_cast<Operator *>(this);
    if (op->right)
      s += op->right->toString();
  }

  return s;
}

void BoolExpression::getVariablesRec(std::set<std::string> &s) {
  if (type == ExpressionType::VARIABLE) {
    SingleCond *cond = static_cast<SingleCond *>(this);
    s.insert(cond->id);
  } else if (type == ExpressionType::AND || type == ExpressionType::OR) {
    Operator *op = static_cast<Operator *>(this);
    if (op->left)
      op->left->getVariablesRec(s);
    if (op->right)
      op->right->getVariablesRec(s);
  }
}

std::set<std::string> BoolExpression::getVariables() {
  std::set<std::string> s;
  getVariablesRec(s);
  return s;
}

//---------Generating Truth Table based on Mintems (SOP Only)-----------

void dynamatic::experimental::boolean::replaceDontCaresRec(
    std::string s, std::set<std::string> &minterms) {
  size_t dontCareIndex = s.find('d');
  if (dontCareIndex == std::string::npos) {
    minterms.insert(s);
    return;
  }
  s[dontCareIndex] = '0';
  replaceDontCaresRec(s, minterms);
  s[dontCareIndex] = '1';
  replaceDontCaresRec(s, minterms);
}

std::set<std::string> dynamatic::experimental::boolean::replaceDontCares(
    const std::set<std::string> &minterms) {
  std::set<std::string> mintermsWithoudDontCares;
  for (StringRef s : minterms)
    replaceDontCaresRec(s.str(), mintermsWithoudDontCares);

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
    std::string &s, std::map<StringRef, int> varIndex) {
  SingleCond *singleCond = static_cast<SingleCond *>(this);
  char &c = s[varIndex[singleCond->id]];
  if (c == 'd')
    c = singleCond->isNegated ? '0' : '1';
  else if (c == '0')
    c = singleCond->isNegated ? '0' : 'n';
  else if (c == '1')
    c = (!singleCond->isNegated) ? '1' : 'n';
}

void BoolExpression::generateMintermAnd(
    std::string &s, const std::map<StringRef, int> &varIndex) {
  if (type == ExpressionType::VARIABLE) {
    generateMintermVariable(s, varIndex);
  } else if (type == ExpressionType::ZERO) { // 0 . exp =0 -> not a a minterm
    s[0] = 'n';
  } else if (type == ExpressionType::AND || type == ExpressionType::OR) {
    Operator *op = static_cast<Operator *>(this);
    if (op->left)
      op->left->generateMintermAnd(s, varIndex);
    if (op->right)
      op->right->generateMintermAnd(s, varIndex);
  }
}

void BoolExpression::generateMintermsOr(
    unsigned numOfVariables, const std::map<StringRef, int> &varIndex,
    std::set<std::string> &minterms) {
  if (type == ExpressionType::OR) {
    Operator *op = static_cast<Operator *>(this);
    if (op->left)
      op->left->generateMintermsOr(numOfVariables, varIndex, minterms);
    if (op->right)
      op->right->generateMintermsOr(numOfVariables, varIndex, minterms);
  } else if (type == ExpressionType::ONE) { // 1 + exp = 1;
    std::string s(numOfVariables, 'd');
    minterms.insert(s);
  } else if (type == ExpressionType::AND) {
    std::string s(numOfVariables, 'd'); // initializing s
    generateMintermAnd(s, varIndex);
    // no null in the minterm -> minterm is valid
    if (s.find('n') == std::string::npos)
      minterms.insert(s);
  }
}

// 1- generate all the minterms with don't cars
// 2- replace the don't cares wit 0s and 1s
// 3- loop over all rows in the truth table and check wether it corresponds to a
// minterm or not
std::set<std::string> BoolExpression::generateTruthTable() {
  std::set<std::string> variables = BoolExpression::getVariables();
  unsigned numOfVariables = variables.size();
  std::map<StringRef, int> varIndex;
  int index = 0;
  for (llvm::StringRef s : variables)
    varIndex[s] = index++;
  // generate the minterms
  std::set<std::string> minterms;
  generateMintermsOr(numOfVariables, varIndex, minterms);
  std::set<std::string> mintermsWithoutDontCares = replaceDontCares(minterms);
  // generate the truth table
  std::string s(numOfVariables, 'd');
  std::set<std::string> r = {s};
  std::set<std::string> rows = replaceDontCares(r);
  std::set<std::string> table;
  for (StringRef row : rows) {
    // if the row corresponds to the minterm, then f is 1. Else f is 0
    if (mintermsWithoutDontCares.find(row.str()) !=
        mintermsWithoutDontCares.end())
      table.insert(row.str() + "1");
    else
      table.insert(row.str() + "0");
  }
  return table;
}

//--------Printing BoolExpresssion--------
static constexpr unsigned PRINTING_SPACE = 10;

// Inspired from https://www.geeksforgeeks.org/print-binary-tree-2-dimensions/
void BoolExpression::print(int space) {
  // Increase distance between levels
  space += PRINTING_SPACE;

  // Process right child first
  if (type == ExpressionType::OR || type == ExpressionType::AND) {
    Operator *op = static_cast<Operator *>(this);
    op->right->print(space);
  }
  // Print current node after space
  // count
  llvm::errs() << "\n";
  for (int i = PRINTING_SPACE; i < space; i++)
    llvm::errs() << " ";

  switch (type) {
  case ExpressionType::VARIABLE: {
    SingleCond *singleCond = static_cast<SingleCond *>(this);
    if (singleCond->isNegated)
      llvm::errs() << "~";
    llvm::errs() << singleCond->id << "\n";
    break;
  }
  case ExpressionType::ONE:
    llvm::errs() << "1 \n";
    break;
  case ExpressionType::ZERO:
    llvm::errs() << "0 \n";
    break;
  case ExpressionType::OR:
    llvm::errs() << "+ \n";
    break;
  case ExpressionType::AND:
    llvm::errs() << ". \n";
    break;
  default:
    break;
  }

  // Process left child
  if (type == ExpressionType::OR || type == ExpressionType::AND) {
    Operator *op = static_cast<Operator *>(this);
    op->left->print(space);
  }
}

//--------------Library Helper Functions--------------

BoolExpression *BoolExpression::propagateNegation(bool negated) {
  if (this->type == ExpressionType::NOT) {
    Operator *op = static_cast<Operator *>(this);
    if (op->right)
      return op->right->propagateNegation(!negated);
  }
  if (negated) {
    if (this->type == ExpressionType::AND) {
      this->type = ExpressionType::OR;
    } else if (this->type == ExpressionType::OR) {
      this->type = ExpressionType::AND;
    } else if (this->type == ExpressionType::VARIABLE) {
      SingleCond *singleCond = static_cast<SingleCond *>(this);
      singleCond->isNegated = !singleCond->isNegated;
    } else if (this->type == ExpressionType::ONE) {
      this->type = ExpressionType::ZERO;
    } else if (this->type == ExpressionType::ZERO) {
      this->type = ExpressionType::ONE;
    }
  }
  if (this->type == ExpressionType::AND || this->type == ExpressionType::OR) {
    Operator *op = static_cast<Operator *>(this);
    if (op->left)
      op->left = op->left->propagateNegation(negated);
    if (op->right)
      op->right = op->right->propagateNegation(negated);
  }
  return this;
}

//--------------Espresso--------------

std::string BoolExpression::runEspresso() {
  std::string espressoInput = "";
  std::set<std::string> vars = this->getVariables();
  // adding the number of inputs and outputs to the file
  espressoInput += (".i " + std::to_string(vars.size()) + "\n");
  espressoInput += ".o 1\n";
  // adding the names of the input variables to the file
  espressoInput += ".ilb ";
  for (const std::string &var : vars)
    espressoInput += (var + " ");
  espressoInput += "\n";
  // add the name of the output f to the file
  espressoInput += ".ob f\n";
  // generate and add the truth table
  std::set<std::string> truthTable = this->generateTruthTable();
  for (const std::string &row : truthTable)
    espressoInput += (row + "\n");
  std::string result = "";
  // char *r = run_espresso(espressoInput.data());
  // std::string result = r;
  if (result == "Failed to Minimize")
    return result;
  int start = result.find('=');
  int end = result.find(';');
  return result.substr(start + 1, end - start - 1);
}

//--------------Library APIs--------------

BoolExpression *BoolExpression::parseSop(llvm::StringRef strSop) {
  Parser parser(strSop);
  BoolExpression *exp = parser.parseSop();
  if (!exp)
    return nullptr;
  return exp->propagateNegation(false);
}

std::string BoolExpression::sopToString() { return this->toString(); }

// returns a dynamically-allocated variable
BoolExpression *BoolExpression::boolVar(std::string id) {
  return new SingleCond(ExpressionType::VARIABLE, std::move(id), false);
}

// returns a dynamically-allocated variable
BoolExpression *BoolExpression::boolAnd(BoolExpression *exp1,
                                        BoolExpression *exp2) {
  return new Operator(ExpressionType::AND, exp1, exp2);
}

// returns a dynamically-allocated variable
BoolExpression *BoolExpression::boolOr(BoolExpression *exp1,
                                       BoolExpression *exp2) {
  return new Operator(ExpressionType::OR, exp1, exp2);
}

BoolExpression *BoolExpression::boolNegate() {
  return this->propagateNegation(true);
}

BoolExpression *BoolExpression::boolMinimize() {
  std::string espressoResult = this->runEspresso();
  // if espresso fails, return the expression as is
  if (espressoResult == "Failed to minimize")
    return this;
  // if espresso returns " ", then f = 0
  if (espressoResult == " ")
    return new BoolExpression(ExpressionType::ZERO);
  // if espresso returns " ()", then f = 1
  if (espressoResult == " ()")
    return new BoolExpression(ExpressionType::ONE);
  return parseSop(espressoResult);
}