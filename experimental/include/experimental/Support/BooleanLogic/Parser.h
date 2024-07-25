//===- Parser.h - Parser for boolean expressions ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structure and methods for parsing boolean logic
// expressions.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_PARSER_H
#define EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_PARSER_H

#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/BooleanLogic/Lexer.h"
#include <stack>
#include <utility>

namespace dynamatic {
namespace experimental {
namespace boolean {

// Represents elements in the parsing stack,
// which can be either expressions or terms (tokens).
struct StackNode {
  BoolExpression *expr;
  std::optional<Token> term;

  // Constructors
  StackNode(Token tt) : expr(nullptr), term(std::move(tt)){};

  StackNode(BoolExpression *e) : expr(e){};

  ~StackNode() {
    if (expr != nullptr)
      delete expr;
  }
  /// Prints a StackNode in case of an error
  void printError();
};

/// The Parser class provides
/// methods for parsing sum-of-products (SOP) expressions using a lexical
/// analyzer to generate tokens and a stack-based approach to construct the
/// parse tree.
class Parser {
public:
  LexicalAnalyzer lexer;

  /// Implements the main parsing logic, using the precedence table
  /// to guide token processing and reduction.
  BoolExpression *parseSop();

  /// Constructor
  Parser(llvm::StringRef s);

private:
  // Ensures the correct terminal node is used for precedence
  // comparison.
  StackNode *terminalPeek(std::vector<StackNode *> stack);
};

/// Creates a new BoolExpression node for AND/OR operations.
/// Returns a dynamically-allocated variable
BoolExpression *constructNodeOperator(StackNode *operate, StackNode *s1,
                                      StackNode *s2);

/// Creates a new BoolExpression node for NOT operations.
/// Returns a dynamically-allocated variable
BoolExpression *constructNodeNegator(StackNode *s1);

/// Wraps a variable token in a BoolExpression node.
/// Returns a dynamically-allocated variable
StackNode *termToExpr(StackNode *s);

/// term for ~, ( , ) , . , +
/// Returns a dynamically-allocated variable
StackNode *constructTermStackNode(Token t);

/// Handles AND/OR operations in the parsing stack.
/// Returns a dynamically-allocated variable
StackNode *constructOperatorStackNode(StackNode *operate, StackNode *s1,
                                      StackNode *s2);

/// Handles NOT operations in the parsing stack.
/// Returns a dynamically-allocated variable
StackNode *constructNegatorStackNode(StackNode *s1);

/// Reduces a sequence of tokens based on operator precedence and returns the
/// resulting stack node.
StackNode *reduce(std::stack<StackNode *> stack);

} // namespace boolean
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_PARSER_H
