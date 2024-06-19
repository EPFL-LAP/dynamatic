//===- Parser.cpp - // Implementation of Parser for Boolean Logic Expressions
//-----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Parser class, which is responsible for parsing a
// tokenized boolean logic expression into a boolean expression tree.
// The parser uses an operator precedence table to manage parsing rules and
// handles logical operators (AND, OR, NOT) and parentheses. The parsing process
// involves constructing nodes for the expression tree and reducing sequences of
// tokens based on precedence rules.
//
//===----------------------------------------------------------------------===//

#include <array>

#include <cassert>

#include <stack>
#include <string>
#include <utility>

#include <vector>

#include "dynamatic/Support/BooleanExpression.h"
#include "dynamatic/Support/Lexer.h"
#include "dynamatic/Support/Parser.h"
#include <cmath>

using namespace dynamatic;

enum class Compare : int { lessThan, greaterThan, equal, error, accept };

// Define the static constexpr precedence table
static constexpr std::array<std::array<Compare, 7>, 7> PRECEDENCE_TABLE = {
    {{Compare::greaterThan, Compare::greaterThan, Compare::greaterThan,
      Compare::lessThan, Compare::greaterThan, Compare::lessThan,
      Compare::greaterThan},
     {Compare::lessThan, Compare::greaterThan, Compare::greaterThan,
      Compare::lessThan, Compare::greaterThan, Compare::lessThan,
      Compare::greaterThan},
     {Compare::lessThan, Compare::lessThan, Compare::greaterThan,
      Compare::lessThan, Compare::greaterThan, Compare::lessThan,
      Compare::greaterThan},
     {Compare::lessThan, Compare::lessThan, Compare::lessThan,
      Compare::lessThan, Compare::equal, Compare::lessThan, Compare::error},
     {Compare::greaterThan, Compare::greaterThan, Compare::greaterThan,
      Compare::error, Compare::greaterThan, Compare::error,
      Compare::greaterThan},
     {Compare::greaterThan, Compare::greaterThan, Compare::greaterThan,
      Compare::error, Compare::greaterThan, Compare::error,
      Compare::greaterThan},
     {Compare::lessThan, Compare::lessThan, Compare::lessThan,
      Compare::lessThan, Compare::error, Compare::lessThan, Compare::accept}}};

// returns a dynamically-allocated variable
BoolExpression *dynamatic::constructNodeOperator(StackNode *operate,
                                                 StackNode *s1, StackNode *s2) {
  assert(operate != nullptr && s1 != nullptr && s2 != nullptr);
  Token t = operate->term;
  ExpressionType oo;
  if (t.tokenType == TokenType::AndToken)
    oo = ExpressionType::And;
  else if (t.tokenType == TokenType::OrToken)
    oo = ExpressionType::Or;
  BoolExpression *e1 = s1->expr;
  BoolExpression *e2 = s2->expr;
  return new BoolExpression(oo, e1, e2);
}

// returns a dynamically-allocated variable
BoolExpression *dynamatic::constructNodeNegator(StackNode *s1) {
  assert(s1 != nullptr);
  return new BoolExpression(ExpressionType::Not, nullptr, s1->expr);
}

// returns a dynamically-allocated variable
StackNode *dynamatic::termToExpr(StackNode *s) {
  assert(s != nullptr);
  return new StackNode(
      StackNodeType::Expr,
      new BoolExpression(ExpressionType::Variable, s->term.lexeme));
}

// returns a dynamically-allocated variable
StackNode *dynamatic::constructTermStackNode(Token t) {
  return new StackNode(StackNodeType::Term, std::move(t));
}

// returns a dynamically-allocated variable
StackNode *dynamatic::constructOperatorStackNode(StackNode *operate,
                                                 StackNode *s1, StackNode *s2) {
  return new StackNode(StackNodeType::Expr,
                       constructNodeOperator(operate, s1, s2));
}

// returns a dynamically-allocated variable
StackNode *dynamatic::constructNegatorStackNode(StackNode *s1) {
  return new StackNode(StackNodeType::Expr, constructNodeNegator(s1));
}

StackNode *dynamatic::reduce(std::stack<StackNode *> stack) {
  // Case 1: Handling parentheses: expr --> ( expr )
  if (stack.size() == 3 &&
      (stack.top()->type == StackNodeType::Term &&
       stack.top()->term.tokenType == TokenType::LparenToken)) {
    stack.pop();
    StackNode *ex = stack.top();
    stack.pop();
    if (stack.top()->type == StackNodeType::Term &&
        stack.top()->term.tokenType == TokenType::RparenToken)
      return ex;
    syntaxError();
    return nullptr;
  }

  // Case 2: Handling binary operators: expr -> expr AND expr || expr OR expr
  if (stack.size() == 3) { // expr -> expr AND expr || expr OR expr
    StackNode *s1 = stack.top();
    stack.pop();
    StackNode *operate = stack.top();
    stack.pop();
    StackNode *s2 = stack.top();
    stack.pop();
    return constructOperatorStackNode(operate, s1, s2);
  }

  // Case 3: Handling negation: expr -> NOT expr
  if (stack.size() == 2 &&
      (stack.top()->type == StackNodeType::Term &&
       stack.top()->term.tokenType == TokenType::NotToken)) {
    stack.pop();
    StackNode *ex = stack.top();
    return constructNegatorStackNode(ex);
  }

  // Case 4: Handling single variable: expr -> variable
  if (stack.size() == 1 &&
      stack.top()->term.tokenType == TokenType::VariableToken) {
    return termToExpr(stack.top());
  }

  // If none of the above cases match, it's a syntax error
  syntaxError();
  return nullptr;
}

Parser::Parser(std::string s) {
  lexer = LexicalAnalyzer(std::move(s));
  lexer.tokenize();
};

StackNode *Parser::terminalPeek(std::vector<StackNode *> stack) {
  if (stack.at(stack.size() - 1)->type == StackNodeType::Term)
    return stack.at(stack.size() - 1);
  if (stack.at(stack.size() - 2)->type == StackNodeType::Term)
    return stack.at(stack.size() - 2);
  syntaxError();
  return nullptr;
}

BoolExpression *Parser::parseSop() {
  std::vector<StackNode *> stack;
  Token start;
  stack.push_back(constructTermStackNode(start));
  while (true) {
    // Peek at the next token from the lexer.
    Token t1 = lexer.peek(1);
    int type1 = static_cast<int>(t1.tokenType);

    // Peek at the top terminal node in the stack.
    StackNode *s2 = terminalPeek(stack);
    if (s2 == nullptr)
      return nullptr;
    Token t2 = s2->term;
    int type2 = static_cast<int>(t2.tokenType);

    // Compare the precedence of the current top token with the next token.
    if (PRECEDENCE_TABLE[type2][type1] == Compare::lessThan ||
        PRECEDENCE_TABLE[type2][type1] == Compare::equal) {
      Token t3 = lexer.getToken();
      stack.push_back(constructTermStackNode(t3));
    } else if (PRECEDENCE_TABLE[type2][type1] == Compare::greaterThan) {
      std::stack<StackNode *> rhs;
      StackNode *lastPoppedTerminal = terminalPeek(stack);

      // Pop nodes from the stack until the precedence condition is met.
      while (stack.at(stack.size() - 1)->type != StackNodeType::Term ||
             PRECEDENCE_TABLE[static_cast<int>(
                 ((stack.at(stack.size() - 1))->term).tokenType)]
                             [static_cast<int>(
                                 (lastPoppedTerminal->term).tokenType)] !=
                 Compare::lessThan) {
        StackNode *s = stack.at(stack.size() - 1);
        stack.pop_back();

        if (s->type == StackNodeType::Term)
          lastPoppedTerminal = s;

        rhs.push(s);
      }
      // Reduce the right-hand side nodes into a single node.
      StackNode *reduced = reduce(rhs);
      if (reduced == nullptr)
        return nullptr;

      stack.push_back(reduced);
    } else if (PRECEDENCE_TABLE[type2][type1] == Compare::accept) {
      return stack.at(1)->expr; // Return the root of the expression tree.
    } else {
      syntaxError();
      return nullptr;
    }
  }
}
