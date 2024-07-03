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

// The StackNode struct represents elements in the parsing stack,
// which can be either expressions or terms (tokens).
struct StackNode {
  BoolExpression *expr;
  Token term;

  // Constructors
  StackNode(Token tt) : term(std::move(tt)){};

  StackNode(BoolExpression *e) : expr(e){};

  ~StackNode() {
    if (expr != nullptr)
      delete expr;
  }

  void printError();
};

// The Parser class provides
// methods for parsing sum-of-products (SOP) expressions using a lexical
// analyzer to generate tokens and a stack-based approach to construct the parse
// tree.
class Parser {
public:
  LexicalAnalyzer lexer;

  // Function that implements the main parsing logic, using the precedence table
  // to guide token processing and reduction.
  BoolExpression *parseSop();

  // Constructor
  Parser(llvm::StringRef s);

private:
  // This function ensures the correct terminal node is used for precedence
  // comparison.
  StackNode *terminalPeek(std::vector<StackNode *> stack);
};

// create a new BoolExpression node for AND/OR operations.
BoolExpression *constructNodeOperator(StackNode *operate, StackNode *s1,
                                      StackNode *s2);

// create a new BoolExpression node for NOT operations.
BoolExpression *constructNodeNegator(StackNode *s1);

// wrap a variable token in a BoolExpression node.
StackNode *termToExpr(StackNode *s);

// term for ~, ( , ) , . , +
StackNode *constructTermStackNode(Token t);

// handle AND/OR operations in the parsing stack.
StackNode *constructOperatorStackNode(StackNode *operate, StackNode *s1,
                                      StackNode *s2);

// handle NOT operations in the parsing stack.
StackNode *constructNegatorStackNode(StackNode *s1);

// reduce a sequence of tokens based on operator precedence and returns the
// resulting stack node.
StackNode *reduce(std::stack<StackNode *> stack);

} // namespace boolean
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_PARSER_H
