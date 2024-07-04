//===- Parser.cpp - Parser for boolean expressions --------------*- C++ -*-===//
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

#include "experimental/Support/BooleanLogic/Parser.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/BooleanLogic/Lexer.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cassert>

using namespace dynamatic::experimental::boolean;

void StackNode::printError() {
  llvm::errs() << "Operator precedence logial error at ";
  if (expr)
    expr->print();
  if (term.has_value())
    llvm::errs() << term.value().lexeme << "\n";
}

enum class Compare { LESS_THAN, GREATER_THAN, EQUAL, ERROR, ACCEPT };

/// Defines the static constexpr precedence tableS
static constexpr std::array<std::array<Compare, 7>, 7> PRECEDENCE_TABLE = {
    {{Compare::GREATER_THAN, Compare::GREATER_THAN, Compare::GREATER_THAN,
      Compare::LESS_THAN, Compare::GREATER_THAN, Compare::LESS_THAN,
      Compare::GREATER_THAN},
     {Compare::LESS_THAN, Compare::GREATER_THAN, Compare::GREATER_THAN,
      Compare::LESS_THAN, Compare::GREATER_THAN, Compare::LESS_THAN,
      Compare::GREATER_THAN},
     {Compare::LESS_THAN, Compare::LESS_THAN, Compare::GREATER_THAN,
      Compare::LESS_THAN, Compare::GREATER_THAN, Compare::LESS_THAN,
      Compare::GREATER_THAN},
     {Compare::LESS_THAN, Compare::LESS_THAN, Compare::LESS_THAN,
      Compare::LESS_THAN, Compare::EQUAL, Compare::LESS_THAN, Compare::ERROR},
     {Compare::GREATER_THAN, Compare::GREATER_THAN, Compare::GREATER_THAN,
      Compare::ERROR, Compare::GREATER_THAN, Compare::ERROR,
      Compare::GREATER_THAN},
     {Compare::GREATER_THAN, Compare::GREATER_THAN, Compare::GREATER_THAN,
      Compare::ERROR, Compare::GREATER_THAN, Compare::ERROR,
      Compare::GREATER_THAN},
     {Compare::LESS_THAN, Compare::LESS_THAN, Compare::LESS_THAN,
      Compare::LESS_THAN, Compare::ERROR, Compare::LESS_THAN,
      Compare::ACCEPT}}};

BoolExpression *dynamatic::experimental::boolean::constructNodeOperator(
    StackNode *operate, StackNode *s1, StackNode *s2) {
  assert((operate && s1 && s2) &&
         "cannot construct operator node with missing operand");
  Token t = operate->term.value();
  ExpressionType oo;
  if (t.tokenType == TokenType::AND_TOKEN)
    oo = ExpressionType::And;
  else
    oo = ExpressionType::Or;
  BoolExpression *e1 = s1->expr;
  BoolExpression *e2 = s2->expr;
  return new Operator(oo, e1, e2);
}

BoolExpression *
dynamatic::experimental::boolean::constructNodeNegator(StackNode *s1) {
  assert(s1 && "cannot negate null node");
  return new Operator(ExpressionType::Not, nullptr, s1->expr);
}

StackNode *dynamatic::experimental::boolean::termToExpr(StackNode *s) {
  assert(s && "cannot convert a null term to an expression");
  ExpressionType t = ExpressionType::Variable;
  if (s->term.value().lexeme == "0")
    t = ExpressionType::Zero;
  if (s->term.value().lexeme == "1")
    t = ExpressionType::One;
  return new StackNode(new SingleCond(t, s->term.value().lexeme));
}

StackNode *dynamatic::experimental::boolean::constructTermStackNode(Token t) {
  return new StackNode(std::move(t));
}

StackNode *dynamatic::experimental::boolean::constructOperatorStackNode(
    StackNode *operate, StackNode *s1, StackNode *s2) {
  return new StackNode(constructNodeOperator(operate, s1, s2));
}

StackNode *
dynamatic::experimental::boolean::constructNegatorStackNode(StackNode *s1) {
  return new StackNode(constructNodeNegator(s1));
}

StackNode *
dynamatic::experimental::boolean::reduce(std::stack<StackNode *> stack) {
  // Case 1: Handling parentheses: expr --> ( expr )
  if (stack.size() == 3 &&
      (!stack.top()->expr &&
       stack.top()->term.value().tokenType == TokenType::LPAREN_TOKEN)) {
    stack.pop();
    StackNode *ex = stack.top();
    stack.pop();
    if (!stack.top()->expr &&
        stack.top()->term.value().tokenType == TokenType::RPAREN_TOKEN)
      return ex;
    stack.top()->printError();
    return nullptr;
  }

  /// Case 2: Handling binary operators: expr -> expr AND expr || expr OR expr
  if (stack.size() == 3) {
    StackNode *s1 = stack.top();
    stack.pop();
    StackNode *operate = stack.top();
    stack.pop();
    StackNode *s2 = stack.top();
    stack.pop();
    return constructOperatorStackNode(operate, s1, s2);
  }

  /// Case 3: Handling negation: expr -> NOT expr
  if (stack.size() == 2 &&
      (!stack.top()->expr &&
       stack.top()->term.value().tokenType == TokenType::NOT_TOKEN)) {
    stack.pop();
    StackNode *ex = stack.top();
    return constructNegatorStackNode(ex);
  }

  /// Case 4: Handling single variable: expr -> variable
  if (stack.size() == 1 &&
      stack.top()->term.value().tokenType == TokenType::VARIABLE_TOKEN) {
    return termToExpr(stack.top());
  }

  /// If none of the above cases match, it's a syntax error
  if (stack.size() > 0)
    stack.top()->printError();
  return nullptr;
}

Parser::Parser(llvm::StringRef s) { lexer = LexicalAnalyzer(s); };

StackNode *Parser::terminalPeek(std::vector<StackNode *> stack) {
  if (!stack.at(stack.size() - 1)->expr)
    return stack.at(stack.size() - 1);
  if (!stack.at(stack.size() - 2)->expr)
    return stack.at(stack.size() - 2);
  stack.at(0)->printError();
  return nullptr;
}

BoolExpression *Parser::parseSop() {
  if (mlir::failed(lexer.tokenize()))
    return nullptr;
  std::vector<StackNode *> stack;
  Token start;
  stack.push_back(constructTermStackNode(start));
  while (true) {
    /// Peek at the next token from the lexer.
    Token t1 = lexer.peek(1);
    int type1 = static_cast<int>(t1.tokenType);

    /// Peek at the top terminal node in the stack.
    StackNode *s2 = terminalPeek(stack);
    if (!s2)
      return nullptr;
    Token t2 = s2->term.value();
    int type2 = static_cast<int>(t2.tokenType);

    /// Compare the precedence of the current top token with the next token.
    if (PRECEDENCE_TABLE[type2][type1] == Compare::LESS_THAN ||
        PRECEDENCE_TABLE[type2][type1] == Compare::EQUAL) {
      Token t3 = lexer.getToken();
      stack.push_back(constructTermStackNode(t3));
    } else if (PRECEDENCE_TABLE[type2][type1] == Compare::GREATER_THAN) {
      std::stack<StackNode *> rhs;
      StackNode *lastPoppedTerminal = terminalPeek(stack);

      /// Pop nodes from the stack until the precedence condition is met.
      while (
          stack.at(stack.size() - 1)->expr ||
          PRECEDENCE_TABLE[static_cast<int>(
              ((stack.at(stack.size() - 1))->term).value().tokenType)]
                          [static_cast<int>(
                              (lastPoppedTerminal->term).value().tokenType)] !=
              Compare::LESS_THAN) {
        StackNode *s = stack.at(stack.size() - 1);
        stack.pop_back();

        if (!s->expr)
          lastPoppedTerminal = s;

        rhs.push(s);
      }
      /// Reduce the right-hand side nodes into a single node.
      StackNode *reduced = reduce(rhs);
      if (!reduced)
        return nullptr;

      stack.push_back(reduced);
    } else if (PRECEDENCE_TABLE[type2][type1] == Compare::ACCEPT) {
      /// Return the root of the expression tree.
      return stack.at(1)->expr;
    } else {
      llvm::errs() << "Operator precedence logial error at " << t1.lexeme
                   << "n";
      return nullptr;
    }
  }
}
