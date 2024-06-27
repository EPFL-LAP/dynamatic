//===- BooleanLogicLibrary.cpp - // Boolean Logic Expression Library
// Implementation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the functions defined in BooleanLogicLibrary.h for
// handling boolean logic expressions.
// It includes propagating negation using DeMorgan's law, and creating basic
// boolean expressions.
// The implemented functions support basic boolean operations such as AND, OR,
// and negation, and enable minimization of boolean expressions using the
// Espresso algorithm.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BooleanLogicLibrary.h"
#include "dynamatic/Support/BooleanExpression.h"
#include "dynamatic/Support/Parser.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

using namespace dynamatic;

BoolExpression *dynamatic::propagateNegation(BoolExpression *root,
                                             bool negated) {
  if (root == nullptr)
    return nullptr;
  if (root->type == ExpressionType::Not) {
    Operator *op = static_cast<Operator *>(root);
    return propagateNegation(op->right, !negated);
  }
  if (negated) {
    if (root->type == ExpressionType::And) {
      root->type = ExpressionType::Or;
    } else if (root->type == ExpressionType::Or) {
      root->type = ExpressionType::And;
    } else if (root->type == ExpressionType::Variable) {
      SingleCond *singleCond = static_cast<SingleCond *>(root);
      singleCond->isNegated = !singleCond->isNegated;
    } else if (root->type == ExpressionType::One) {
      root->type = ExpressionType::Zero;
    } else if (root->type == ExpressionType::Zero) {
      root->type = ExpressionType::One;
    }
  }
  if (root->type == ExpressionType::And || root->type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(root);
    op->left = propagateNegation(op->left, negated);
    op->right = propagateNegation(op->right, negated);
  }
  return root;
}

//--------------Espresso--------------

std::string dynamatic::runEspresso(BoolExpression *expr) {
  std::string espressoInput = "";
  std::set<std::string> vars = expr->getVariables();
  // adding the number of inputs and outputs to the file
  espressoInput += (".i " + std::to_string(vars.size()) + "\n");
  espressoInput += ".o 1\n";
  // adding the names of the input variables to the file
  espressoInput += ".ilb ";
  for (const std::string &var : vars) {
    espressoInput += (var + " ");
  }
  espressoInput += "\n";
  // add the name of the output f to the file
  espressoInput += ".ob f\n";
  // generate and add the truth table
  std::set<std::string> truthTable = expr->generateTruthTable();
  for (const std::string &row : truthTable) {
    espressoInput += (row + "\n");
  }
  std::string result = "";
  // char *r = run_espresso(espressoInput.data());
  // std::string result = r;
  if (result == "Failed to Minimize")
    return result;
  int start = result.find('=');
  int end = result.find(';');
  return result.substr(start + 1, end - start - 1);
}

BoolExpression *dynamatic::parseSop(std::string strSop) {
  Parser parser(std::move(strSop));
  return propagateNegation(parser.parseSop(), false);
}

std::string dynamatic::sopToString(BoolExpression *exp1) {
  return exp1->toString();
}

// returns a dynamically-allocated variable
BoolExpression *dynamatic::boolVar(std::string id) {
  return new SingleCond(ExpressionType::Variable, std::move(id), false);
}

// returns a dynamically-allocated variable
BoolExpression *dynamatic::boolAnd(BoolExpression *exp1, BoolExpression *exp2) {
  return new Operator(ExpressionType::And, exp1, exp2);
}

// returns a dynamically-allocated variable
BoolExpression *dynamatic::boolOr(BoolExpression *exp1, BoolExpression *exp2) {
  return new Operator(ExpressionType::Or, exp1, exp2);
}

BoolExpression *dynamatic::boolNegate(BoolExpression *exp1) {
  return propagateNegation(exp1, true);
}

BoolExpression *dynamatic::boolMinimize(BoolExpression *expr) {
  std::string espressoResult = runEspresso(expr);
  // if espresso fails, return the expression as is
  if (espressoResult == "Failed to minimize")
    return expr;
  // if espresso returns " ", then f = 0
  if (espressoResult == " ") {
    return new BoolExpression(ExpressionType::Zero);
  }
  // if espresso returns " ()", then f = 1
  if (espressoResult == " ()")
    return new BoolExpression(ExpressionType::One);
  return (parseSop(espressoResult));
}
