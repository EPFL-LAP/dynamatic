#include <string>
#include <iostream>
#include <vector>

#include "lexer.h"

using namespace std;


//Initiates the lexical analyzer and tokenizes the string
LexicalAnalyzer::LexicalAnalyzer(string s){
    int position = 0;
    while (position < s.length()){
        char current = s[position];
        if(isspace(current)){
            position++;
            continue;
        }else if(current == 'c'){ //variable
            int id_index = position+1;
            while(id_index < s.length() && isdigit(s[id_index])){
                id_index++;
            }
            string id = s.substr(position+1, id_index-position-1);
            if(id.empty()){
                syntax_error();
            }else{
                Token token(id, VARIABLE_TOKEN);
                token_list.push_back(token);
                position = id_index;
            }           
        }else{
            Token token;
            string str(1, current);
            token.lexeme=str;
            switch (current){
                case '.':   token.token_type = AND_TOKEN; break;
                case '+':   token.token_type = OR_TOKEN; break;
                case '~':   token.token_type = NOT_TOKEN; break;
                case '(':   token.token_type = LPAREN_TOKEN; break;
                case ')':   token.token_type = RPAREN_TOKEN; break;
                default:
                    syntax_error();
            }
            token_list.push_back(token);
            position++;
        }   
    }
    Token end_token;
    token_list.push_back(end_token);
};


// GetToken() accesses tokens from the token_list 
Token LexicalAnalyzer::GetToken(){
    Token token;
    if(index < token_list.size()){
        token = token_list[index];
        index = index +1 ;
    }
    return token;
}

// peek(int i) shows the token that is i places far from the current index
Token LexicalAnalyzer::peek(int i){
    if (i <= 0) {      // peeking backward or in place is not allowed
        exit(1);
    }
    Token token;
    int peek_index = index + i-1;
    if(peek_index < token_list.size()){
        token = token_list[peek_index];
    }
    return token;
}