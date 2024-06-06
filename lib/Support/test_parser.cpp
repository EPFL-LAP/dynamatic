#include "parser.h"
#include "parser.cpp"
#include "lexer.h"
#include "lexer.cpp"

int main(){
    //c1. (c2.c1)+~c3 . c1 + c4. ~c1
    Parser parser("c1. (c2.c1)+~c3 . c1 + c4. ~c1");
    BoolExpression* expr = parser.parse_sop();
    printBoolExpressionTree(expr);
    return 0;
}