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
  if (root->type == ExpressionType::Not)
    return propagateNegation(root->right, !negated);
  if (negated) {
    if (root->type == ExpressionType::And)
      root->type = ExpressionType::Or;
    else if (root->type == ExpressionType::Or)
      root->type = ExpressionType::And;
    else if (root->type == ExpressionType::Variable)
      root->singleCond.isNegated = !root->singleCond.isNegated;
  }
  if (root->type != ExpressionType::Variable) {
    root->left = propagateNegation(root->left, negated);
    root->right = propagateNegation(root->right, negated);
  }
  return root;
}

//--------------Espresso--------------

std::string dynamatic::execute(const std::string &command,
                               const std::string &input) {
  int inputPipe[2];
  int outputPipe[2];
  pid_t pid;
  char buffer[128];
  std::string result = "";

  // Create pipes for input and output
  if (pipe(inputPipe) == -1 || pipe(outputPipe) == -1) {
    return "pipe failed!";
  }

  // Fork the process
  pid = fork();
  if (pid == -1) {
    return "Failed to minimize";
  }

  if (pid == 0) { // Child process
    // Redirect stdin to read from input_pipe
    dup2(inputPipe[0], STDIN_FILENO);
    // Redirect stdout to write to output_pipe
    dup2(outputPipe[1], STDOUT_FILENO);
    // Redirect stderr to write to output_pipe
    dup2(outputPipe[1], STDERR_FILENO);

    // Close unused pipe ends
    close(inputPipe[1]);
    close(outputPipe[0]);

    // Execute the command
    execl("/bin/sh", "sh", "-c", command.c_str(), nullptr);

    // If execl fails
    return "Failed to minimize";
  } else { // Parent process
    // Close unused pipe ends
    close(inputPipe[0]);
    close(outputPipe[1]);

    // Write input to the child process
    write(inputPipe[1], input.c_str(), input.length());
    close(inputPipe[1]);

    // Read output from the child process
    ssize_t bytesRead;
    while ((bytesRead = read(outputPipe[0], buffer, sizeof(buffer) - 1)) > 0) {
      buffer[bytesRead] = '\0';
      result += buffer;
    }
    close(outputPipe[0]);

    // Wait for the child process to finish
    waitpid(pid, nullptr, 0);
  }

  return result;
}

/*
Espresso's Input Format:
.i [d]                   # number of input variables
.o 1                     # number of outputs
.ilb [s1] [s2] ... [sn]  # names of input variables
.ob f                    # output name
---truth table rows---
.re

Espresso's Output Format:
f = (minimized function);

Example: x.y.x
Input:
.i 2
.o 1
.ilb x y
.ob f
0 0 0
0 1 0
1 0 0
1 1 1
.e

Output: f = x&y;
*/
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
  std::string result = execute("./espresso -o eqntott ", espressoInput);
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
  return new BoolExpression(ExpressionType::Variable, std::move(id), false);
}

// returns a dynamically-allocated variable
BoolExpression *dynamatic::boolAnd(BoolExpression *exp1, BoolExpression *exp2) {
  return new BoolExpression(ExpressionType::And, exp1, exp2);
}

// returns a dynamically-allocated variable
BoolExpression *dynamatic::boolOr(BoolExpression *exp1, BoolExpression *exp2) {
  return new BoolExpression(ExpressionType::Or, exp1, exp2);
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
