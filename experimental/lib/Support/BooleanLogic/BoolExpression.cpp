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
#include "dynamatic/Support/Espresso/main.h"
#include "experimental/Support/BooleanLogic/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <set>
#include <string>

using namespace dynamatic::experimental::boolean;
using namespace llvm;

//----------BoolExpression Functions Implementatins--------

std::string BoolExpression::toString() {
  std::string s;

  if (type == ExpressionType::And || type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(this);
    if (op->left)
      s += op->left->toString();
  }

  switch (type) {
  case ExpressionType::Variable: {
    SingleCond *singleCond = static_cast<SingleCond *>(this);
    if (singleCond->isNegated)
      s += "~";
    s += singleCond->id;
    break;
  }
  case ExpressionType::One:
    s += "1";
    break;
  case ExpressionType::Zero:
    s += "0";
    break;
  case ExpressionType::Or:
    s += " + ";
    break;
  case ExpressionType::And:
    s += " . ";
    break;
  default:
    break;
  }

  if (type == ExpressionType::And || type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(this);
    if (op->right)
      s += op->right->toString();
  }

  return s;
}

void BoolExpression::getVariablesRec(std::set<std::string> &s) {
  if (type == ExpressionType::Variable) {
    SingleCond *cond = static_cast<SingleCond *>(this);
    s.insert(cond->id);
  } else if (type == ExpressionType::And || type == ExpressionType::Or) {
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

//---------Generating Truth Table For any Boolean Expression-----------

bool BoolExpression::evaluate(std::map<std::string, bool> row) {
  switch (type) {
  case (ExpressionType::Variable): {
    SingleCond *singleCond = static_cast<SingleCond *>(this);
    return (singleCond->isNegated && !(row[singleCond->id])) ||
           (!(singleCond->isNegated) && row[singleCond->id]);
    break;
  }
  case (ExpressionType::One):
    return true;
  case (ExpressionType::Zero):
    return false;
  case (ExpressionType::Or): {
    Operator *op = static_cast<Operator *>(this);
    bool left = false;
    if (op->left)
      left = op->left->evaluate(row);
    bool right = false;
    if (op->right)
      right = op->right->evaluate(row);
    return left || right;
  }
  case (ExpressionType::And): {
    Operator *op = static_cast<Operator *>(this);
    bool left = true;
    if (op->left)
      left = op->left->evaluate(row);
    bool right = true;
    if (op->right)
      right = op->right->evaluate(row);
    return left && right;
  }
  case (ExpressionType::Not): {
    Operator *op = static_cast<Operator *>(this);
    return !op->right->evaluate(row);
  }

  default:
    return false;
  }
}

std::vector<std::tuple<std::map<std::string, bool>, bool>>
BoolExpression::generateTruthTable() {
  std::vector<std::tuple<std::map<std::string, bool>, bool>> truthTable;
  std::set<std::string> variables = getVariables();
  int numOfRows = pow(2, variables.size());
  for (int i = 0; i < numOfRows; i++) {
    std::map<std::string, bool> row;

    // Iterate over the variables and set their truth values based on the
    // current combination
    int j = 0;
    for (const std::string &variable : variables) {
      // Extract the truth value of the current variable from the bit
      // representation of 'i'
      bool truthValue = (i >> j) & 1;
      // Map the variable to its truth value in the current row
      row[variable] = truthValue;
      j++;
    }
    bool res = evaluate(row);
    truthTable.emplace_back(row, res);
  }
  return truthTable;
}

bool BoolExpression::containsMintern(const std::string &toSearch) {
  if (type == ExpressionType::And || type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(this);
    return op->left->containsMintern(toSearch) ||
           op->right->containsMintern(toSearch);
  }

  if (type == ExpressionType::Variable) {
    SingleCond *singleCond = static_cast<SingleCond *>(this);
    return toSearch == singleCond->id;
  }
  return false;
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
  if (type == ExpressionType::Variable) {
    generateMintermVariable(s, varIndex);
  } else if (type == ExpressionType::Zero) {
    // 0 . exp =0 -> not a a minterm
    s[0] = 'n';
  } else if (type == ExpressionType::And || type == ExpressionType::Or) {
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
  if (type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(this);
    if (op->left)
      op->left->generateMintermsOr(numOfVariables, varIndex, minterms);
    if (op->right)
      op->right->generateMintermsOr(numOfVariables, varIndex, minterms);
  } else if (type == ExpressionType::One) {
    // 1 + exp = 1;
    std::string s(numOfVariables, 'd');
    minterms.insert(s);
  } else if (type == ExpressionType::And) {
    std::string s(numOfVariables, 'd');
    generateMintermAnd(s, varIndex);
    // no null in the minterm -> minterm is valid
    if (s.find('n') == std::string::npos)
      minterms.insert(s);
  }
}

std::set<std::string> BoolExpression::generateTruthTableSop() {
  std::set<std::string> variables = BoolExpression::getVariables();
  unsigned numOfVariables = variables.size();
  std::map<StringRef, int> varIndex;
  int index = 0;
  for (llvm::StringRef s : variables)
    varIndex[s] = index++;
  // generate all the minterms with don't cars
  std::set<std::string> minterms;
  generateMintermsOr(numOfVariables, varIndex, minterms);
  // replace the don't cares wit 0s and 1s
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

void BoolExpression::print(int space) {
  // Increase distance between levels
  space += PRINTING_SPACE;

  // Process right child first
  if (type == ExpressionType::Or || type == ExpressionType::And) {
    Operator *op = static_cast<Operator *>(this);
    op->right->print(space);
  }

  llvm::errs() << "\n";
  for (int i = PRINTING_SPACE; i < space; i++)
    llvm::errs() << " ";

  switch (type) {
  case ExpressionType::Variable: {
    SingleCond *singleCond = static_cast<SingleCond *>(this);
    if (singleCond->isNegated)
      llvm::errs() << "~";
    llvm::errs() << singleCond->id << "\n";
    break;
  }
  case ExpressionType::One:
    llvm::errs() << "1 \n";
    break;
  case ExpressionType::Zero:
    llvm::errs() << "0 \n";
    break;
  case ExpressionType::Or:
    llvm::errs() << "+ \n";
    break;
  case ExpressionType::And:
    llvm::errs() << ". \n";
    break;
  default:
    break;
  }

  // Process left child
  if (type == ExpressionType::Or || type == ExpressionType::And) {
    Operator *op = static_cast<Operator *>(this);
    op->left->print(space);
  }
}

//--------------Library Helper Functions--------------

BoolExpression *BoolExpression::propagateNegation(bool negated) {
  if (this->type == ExpressionType::Not) {
    Operator *op = static_cast<Operator *>(this);
    if (op->right)
      return op->right->propagateNegation(!negated);
  }
  if (negated) {
    if (this->type == ExpressionType::And) {
      this->type = ExpressionType::Or;
    } else if (this->type == ExpressionType::Or) {
      this->type = ExpressionType::And;
    } else if (this->type == ExpressionType::Variable) {
      SingleCond *singleCond = static_cast<SingleCond *>(this);
      singleCond->isNegated = !singleCond->isNegated;
    } else if (this->type == ExpressionType::One) {
      this->type = ExpressionType::Zero;
    } else if (this->type == ExpressionType::Zero) {
      this->type = ExpressionType::One;
    }
  }
  if (this->type == ExpressionType::And || this->type == ExpressionType::Or) {
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
  std::vector<std::tuple<std::map<std::string, bool>, bool>> truthTable =
      generateTruthTable();
  for (std::tuple<std::map<std::string, bool>, bool> row : truthTable) {
    std::map<std::string, bool> value = std::get<0>(row);
    for (const std::string &s : vars) {
      espressoInput += (std::string(value[s] ? "1" : "0") + " ");
    }
    bool res = std::get<1>(row);
    espressoInput += (std::string(res ? "1" : "0") + "\n");
  }
  // run espresso
  char *r = run_espresso(espressoInput.data());
  std::string result = r;
  if (result == "Failed to Minimize")
    return result;
  int start = result.find('=');
  int end = result.find(';');
  return result.substr(start + 1, end - start - 1);
}

std::string BoolExpression::runEspressoSop() {
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
  std::set<std::string> truthTable = this->generateTruthTableSop();
  for (const std::string &row : truthTable)
    espressoInput += (row + "\n");
  char *r = run_espresso(espressoInput.data());
  std::string result = r;
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

BoolExpression *BoolExpression::boolZero() {
  return new SingleCond(ExpressionType::Zero, "0", false);
}

BoolExpression *BoolExpression::boolOne() {
  return new SingleCond(ExpressionType::One, "1", false);
}

BoolExpression *BoolExpression::boolVar(std::string id) {
  return new SingleCond(ExpressionType::Variable, std::move(id), false);
}

BoolExpression *BoolExpression::boolAnd(BoolExpression *exp1,
                                        BoolExpression *exp2) {
  return new Operator(ExpressionType::And, exp1, exp2);
}

BoolExpression *BoolExpression::boolOr(BoolExpression *exp1,
                                       BoolExpression *exp2) {
  return new Operator(ExpressionType::Or, exp1, exp2);
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
    return new BoolExpression(ExpressionType::Zero);
  // if espresso returns " ()", then f = 1
  if (espressoResult == " ()")
    return new BoolExpression(ExpressionType::One);
  return parseSop(espressoResult);
}

BoolExpression *BoolExpression::boolMinimizeSop() {
  std::string espressoResult = this->runEspressoSop();
  // if espresso fails, return the expression as is
  if (espressoResult == "Failed to minimize")
    return this;
  // if espresso returns " ", then f = 0
  if (espressoResult == " ")
    return new BoolExpression(ExpressionType::Zero);
  // if espresso returns " ()", then f = 1
  if (espressoResult == " ()")
    return new BoolExpression(ExpressionType::One);
  return parseSop(espressoResult);
}

BoolExpression *BoolExpression::deepCopy() const {
  switch (type) {
  case ExpressionType::And:
  case ExpressionType::Or:
  case ExpressionType::Not: {
    const Operator *op = static_cast<const Operator *>(this);
    return new Operator(type, op->left->deepCopy(),
                        op->right ? op->right->deepCopy() : nullptr);
  }
  case ExpressionType::Variable:
  case ExpressionType::Zero:
  case ExpressionType::One: {
    const SingleCond *sc = static_cast<const SingleCond *>(this);
    return new SingleCond(type, sc->id, sc->isNegated);
  }
  case ExpressionType::End:
    return new SingleCond(ExpressionType::End);
  }
}
