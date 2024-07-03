//===- Lexer.h - Lexer for boolean expressions ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structure for handling lexical analysis,
// specifically for a set of tokens used in boolean logic expressions. The
// LexicalAnalyzer class provides methods for initializing with an input string,
// retrieving the next token, and peeking ahead in the token stream.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_LEXER_H
#define EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_LEXER_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <utility>
#include <vector>

namespace dynamatic {
namespace experimental {
namespace boolean {

/// Enumeration defining the types of tokens in the boolean logic expressions.
enum class TokenType {
  NOT_TOKEN,
  AND_TOKEN,
  OR_TOKEN,
  LPAREN_TOKEN,
  RPAREN_TOKEN,
  VARIABLE_TOKEN,
  END_TOKEN
};

/// Represents an individual token, consisting of a lexeme and a token
/// type.
struct Token {
  /// The string representation of the token.
  std::string lexeme;
  /// The type of the token.
  TokenType tokenType;

  /// Default constructor initializing lexeme to empty string and token type to
  /// EndToken.
  Token() : tokenType(TokenType::END_TOKEN) {}

  /// Constructor initializing lexeme and token type.
  Token(std::string l, TokenType r) : lexeme(std::move(l)), tokenType(r) {}
};

/// Reports syntax errors encountered during lexical analysis.
void syntaxError(char current);

/// Class representing the Lexical Analyzer for boolean logic expressions.
class LexicalAnalyzer {
public:
  /// Constructor to initialize the lexical analyzer with an input string.
  LexicalAnalyzer(llvm::StringRef exp = "");

  /// Processes a string expression, skipping whitespace and
  /// identifying variable tokens from letter-digit sequences. It creates tokens
  /// for operators and parentheses, storing all tokens in tokenList. An end
  /// token is added to mark the completion of tokenization.
  mlir::LogicalResult tokenize();

  /// Retrieves the next token from the token list.
  Token getToken();

  /// Peeks ahead in the token stream by 'i' places.
  Token peek(unsigned int i);

private:
  // List of tokens extracted from the input expression.
  std::vector<Token> tokenList;
  // Current index in the token list.
  size_t index = 0;
  std::string expression;
};

} // namespace boolean
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_BOOLEANLOGIC_LEXER_H
