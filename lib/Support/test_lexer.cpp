#include <iostream>
#include "lexer.h"
#include "lexer.cpp"

using namespace std;

int main(){
    //c1. c2.c1+~c3 . c1 + c4. ~c1
    LexicalAnalyzer lexer("c1. c2.c1+~c3 . c1 + c4. ~c1");
    Token token = lexer.GetToken();
    cout << "Lexeme: " << token.lexeme <<  " of token type "<< token.token_type << endl;
    while(token.token_type != END_TOKEN){
        token = lexer.GetToken();
        cout << "Lexeme: " << token.lexeme <<  " of token type "<< token.token_type << endl;
    }
    return 0;

}