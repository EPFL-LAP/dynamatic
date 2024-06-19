//===- Lexer.cpp - Implementation of Lexical Analyzer for Boolean Logic
// Expressions -----*- C++ -*-===//
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

#include <iostream>

#include <string>

#include <string>

#include <utility>

#include <vector>

#include "dynamatic/Support/Lexer.h"

using namespace llvm;
using namespace dynamatic;

// Function for reporting syntax errors encountered during lexical analysis.
void dynamatic::syntaxError() {
  llvm::errs() << "Syntax error in expression!\n";
}

LexicalAnalyzer::LexicalAnalyzer(std::string exp) {
  expression = std::move(exp);
}

// The function tokenize processes a string expression, skipping whitespace and
// identifying variable tokens from letter-digit sequences. It creates tokens
// for operators and parentheses, storing all tokens in tokenList. An end token
// is added to mark the completion of tokenization.
void LexicalAnalyzer::tokenize() {
  int position = 0;
  while (position < expression.length()) {
    char current = expression[position];
    if (isspace(current)) {
      position++;
      continue;
    }
    if (isalpha(current)) { // If the current character is a letter, it
                            // indicates a variable.
      int idIndex = position + 1;
      while (idIndex < expression.length() && isdigit(expression[idIndex]))
        idIndex++;
      std::string id = expression.substr(position, idIndex - position);
      Token token(id, TokenType::VariableToken);
      tokenList.push_back(token);
      position = idIndex;

    } else {
      Token token;
      std::string str(1, current);
      token.lexeme = str;
      switch (current) {
      case '.':
      case '&':
        token.tokenType = TokenType::AndToken;
        break;
      // case '&':   token.tokenType = TokenType::AndToken; break;
      case '+':
      case '|':
        token.tokenType = TokenType::OrToken;
        break;
      // case '|':   token.tokenType = TokenType::OrToken; break;
      case '~':
      case '!':
        token.tokenType = TokenType::NotToken;
        break;
      // case '!':   token.tokenType = TokenType::NotToken; break;
      case '(':
        token.tokenType = TokenType::LparenToken;
        break;
      case ')':
        token.tokenType = TokenType::RparenToken;
        break;
      default:
        syntaxError();
      }
      tokenList.push_back(token);
      position++;
    }
  }
  Token endToken;
  tokenList.push_back(endToken);
}

Token LexicalAnalyzer::getToken() {
  Token token;
  if (index < tokenList.size()) {
    token = tokenList[index];
    index = index + 1;
  }
  return token;
}

Token LexicalAnalyzer::peek(unsigned int i) {
  Token token;
  int peekIndex = index + i - 1;
  if (peekIndex < tokenList.size())
    token = tokenList[peekIndex];
  return token;
}
