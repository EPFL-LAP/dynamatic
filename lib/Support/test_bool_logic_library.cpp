#include <iostream>
#include "bool_logic_library.cpp"

using namespace std;

int main(){
    string test = "c1. (c2.c1)+~c3 . c1 + c4. ~c1";
    cout << "In string format:" << test << endl;
    BoolExpression * expr = parse_sop(test);
    cout << "In tree format:" << endl;
    printBoolExpressionTree(expr);
    string s = sop_to_string(expr);
    cout << "Back to string format:" << s << endl;
    BoolExpression*  expr2 = propagarteNegation(expr, false);
    cout << "Negated string format:" << sop_to_string(expr2) << endl;
    BoolExpression * test_and1 = parse_sop("c1");
    BoolExpression * test_and2 = parse_sop("c2.~c3");
    BoolExpression*  and_expr = Bool_And(test_and1, test_and2);
    cout << "And format:" << sop_to_string(and_expr) << endl;
    BoolExpression*  or_expr = Bool_Or(test_and1, test_and2);
    cout << "Or format:" << sop_to_string(or_expr) << endl;

    return 0;
}



