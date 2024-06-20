//===- Parser.h - Parser for boolean expressions -----*- C++ -*-===//
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

#ifndef DYNAMATIC_SUPPORT_PARSER_H
#define DYNAMATIC_SUPPORT_PARSER_H

#include "dynamatic/Support/BooleanExpression.h"
#include "dynamatic/Support/Lexer.h"

#include <stack>
#include <utility>

namespace dynamatic {

enum class StackNodeType : int { Expr, Term };

// The StackNode struct represents elements in the parsing stack,
// which can be either expressions or terms (tokens).
struct StackNode {
  StackNodeType type;
  union {
    BoolExpression *expr;
    Token term;
  };

  // Constructors
  StackNode() : type(StackNodeType::Expr), expr(nullptr){};

  StackNode(StackNodeType t, Token tt) : type(t), term(std::move(tt)){};

  StackNode(StackNodeType t, BoolExpression *e) : type(t), expr(e){};

  ~StackNode() {
    if (type == StackNodeType::Expr && expr != nullptr)
      delete expr;
  }
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
  Parser(std::string s);

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
} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_PARSER_H
