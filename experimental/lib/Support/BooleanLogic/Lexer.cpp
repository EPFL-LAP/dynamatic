//===- Lexer.cpp - Lexer for boolean expressions ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LexicalAnalyzer class for tokenizing a string
// representing a boolean logic expression. The constructor processes the input
// string, identifying tokens and storing them in a list. The GetToken method
// retrieves the next token, while peek allows looking ahead in the token
// stream.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/BooleanLogic/Lexer.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic::experimental::boolean;

// Function for reporting syntax errors encountered during lexical analysis.
void dynamatic::experimental::boolean::syntaxError(char current) {
  llvm::errs() << "Syntax error in expression! Invalid character " << current
               << '\n';
}

LexicalAnalyzer::LexicalAnalyzer(StringRef exp) : expression(exp) {}

// The function tokenize processes a string expression, skipping whitespace and
// identifying variable tokens from letter-digit sequences. It creates tokens
// for operators and parentheses, storing all tokens in tokenList. An end token
// is added to mark the completion of tokenization.
mlir::LogicalResult LexicalAnalyzer::tokenize() {
  size_t position = 0;
  while (position < expression.length()) {
    char current = expression[position];
    if (isspace(current)) {
      position++;
      continue;
    }
    if (isalpha(current)) { // If the current character is a letter, it
                            // indicates a variable.
      size_t idIndex = position + 1;
      while (idIndex < expression.length() && isdigit(expression[idIndex]))
        idIndex++;
      std::string id = expression.substr(position, idIndex - position);
      Token token(id, TokenType::VARIABLE_TOKEN);
      tokenList.push_back(token);
      position = idIndex;

    } else {
      Token token;
      std::string str(1, current);
      token.lexeme = str;
      switch (current) {
      case '.':
      case '&':
        token.tokenType = TokenType::AND_TOKEN;
        break;
      case '+':
      case '|':
        token.tokenType = TokenType::OR_TOKEN;
        break;
      case '~':
      case '!':
        token.tokenType = TokenType::NOT_TOKEN;
        break;
      case '(':
        token.tokenType = TokenType::LPAREN_TOKEN;
        break;
      case ')':
        token.tokenType = TokenType::RPAREN_TOKEN;
        break;
      case '0':
      case '1':
        token.tokenType = TokenType::VARIABLE_TOKEN;
        break;
      default:
        syntaxError(current);
        return failure();
      }
      tokenList.push_back(token);
      position++;
    }
  }
  Token endToken;
  tokenList.push_back(endToken);
  return success();
}

Token LexicalAnalyzer::getToken() {
  Token token;
  if (index < tokenList.size()) {
    token = tokenList[index];
    ++index;
  }
  return token;
}

Token LexicalAnalyzer::peek(unsigned int i) {
  Token token;
  size_t peekIndex = index + i - 1;
  if (peekIndex < tokenList.size())
    token = tokenList[peekIndex];
  return token;
}
