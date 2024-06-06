#ifndef __LEXER__H__
#define __LEXER__H__


#include <vector>
#include <string>
#include <iostream>

using namespace std;

enum TokenType{VARIABLE_TOKEN, AND_TOKEN, OR_TOKEN, NOT_TOKEN, LPAREN_TOKEN, RPAREN_TOKEN, END_TOKEN}; 

struct Token {
    string lexeme;
    TokenType token_type;

    Token() : lexeme(""), token_type(END_TOKEN) {};

    Token(string l, TokenType r): lexeme(l), token_type(r) {};
};

void syntax_error() {
	cout << "Syntax error!";
	exit(1);
}


class LexicalAnalyzer {
public:
    LexicalAnalyzer(string s);
    Token GetToken();
    Token peek(int);
    
private:
    vector<Token> token_list;
    int index = 0;
};

#endif