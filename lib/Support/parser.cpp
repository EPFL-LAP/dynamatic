#include <vector>
#include <stack>
#include <queue>
#include <iostream>

#include "parser.h"
#include "lexer.h"

using namespace std;

int Parser::hash(Token tt) {
	TokenType t = tt.token_type;
	if (t== NOT_TOKEN) {
		return 0;
	}
	else if (t == AND_TOKEN) {
		return 1;
	}
	else if (t == OR_TOKEN) {
		return 2;
	}
	else if (t == LPAREN_TOKEN) {
		return 3;
	}
	else if (t == RPAREN_TOKEN) {
		return 4;
	}
	else if (t == VARIABLE_TOKEN) {
		return 5;
	} 
	else{ //END_TOKEN
		return 6;
	}
}

enum comp {
	lessThan, greaterThan, equall, error, accept
};
comp precedanceTable[7][7];

void buildOperatorPrecedenceTable() {
    precedanceTable[0][0] = greaterThan;
	precedanceTable[0][1] = greaterThan;
	precedanceTable[0][2] = greaterThan;
	precedanceTable[0][3] = lessThan;
	precedanceTable[0][4] = greaterThan;
	precedanceTable[0][5] = lessThan;
	precedanceTable[0][6] = greaterThan;

    precedanceTable[1][0] = lessThan;
	precedanceTable[1][1] = greaterThan;
	precedanceTable[1][2] = greaterThan;
	precedanceTable[1][3] = lessThan;
	precedanceTable[1][4] = greaterThan;
	precedanceTable[1][5] = lessThan;
	precedanceTable[1][6] = greaterThan;

    precedanceTable[2][0] = lessThan;
	precedanceTable[2][1] = lessThan;
	precedanceTable[2][2] = greaterThan;
	precedanceTable[2][3] = lessThan;
	precedanceTable[2][4] = greaterThan;
	precedanceTable[2][5] = lessThan;
	precedanceTable[2][6] = greaterThan;

    precedanceTable[3][0] = greaterThan;
	precedanceTable[3][1] = lessThan;
	precedanceTable[3][2] = lessThan;
	precedanceTable[3][3] = lessThan;
	precedanceTable[3][4] = equall;
	precedanceTable[3][5] = lessThan;
	precedanceTable[3][6] = error;

    precedanceTable[4][0] = greaterThan;
	precedanceTable[4][1] = greaterThan;
	precedanceTable[4][2] = greaterThan;
	precedanceTable[4][3] = error;
	precedanceTable[4][4] = greaterThan;
	precedanceTable[4][5] = error;
	precedanceTable[4][6] = greaterThan;

    precedanceTable[5][0] = error;
	precedanceTable[5][1] = greaterThan;
	precedanceTable[5][2] = greaterThan;
	precedanceTable[5][3] = error;
	precedanceTable[5][4] = greaterThan;
	precedanceTable[5][5] = error;
	precedanceTable[5][6] = greaterThan;

    precedanceTable[6][0] = lessThan;
	precedanceTable[6][1] = lessThan;
	precedanceTable[6][2] = lessThan;
	precedanceTable[6][3] = lessThan;
	precedanceTable[6][4] = error;
	precedanceTable[6][5] = lessThan;
	precedanceTable[6][6] = accept;

}

BoolExpression * constructNodeOperator(StackNode* operate, StackNode* s1, StackNode*s2){
    if(operate!= NULL && s1!=NULL && s2!=NULL){
        Token t = operate->term;
        expressionType oo;
        if(t.token_type == AND_TOKEN){
            oo=AND;
        }else if(t.token_type == OR_TOKEN){
            oo=OR;
        }
        BoolExpression* e1 = s1->expr;
	    BoolExpression* e2 = s2->expr;
        return new BoolExpression(oo, e1, e2);
    }else{
        syntax_error();
		return new BoolExpression(END);
    }
}

BoolExpression * constructNodeNegator(StackNode* s1){
    if(s1!=NULL){
        BoolExpression* e1 = s1->expr;
        return new BoolExpression(NOT, nullptr, e1);
    }else{
        syntax_error();
		return new BoolExpression(END);
    }
}

//turing a SingleCond from term to expr
StackNode* TERM_TO_EXP(StackNode* s) {	
    if(s!=NULL){
	    return new StackNode(EXPR, new BoolExpression(VARIABLE, stoi(s->term.lexeme)));
	}else{
		syntax_error();
		return new StackNode();
	}
}

//term for ~, ( , ) , . , + 
StackNode* construct_TERM_Stack_Node(Token t) {
	return new StackNode(TERM, t);
}

StackNode* construct_Operator_Stack_Node(StackNode* operate,  StackNode* s1,  StackNode* s2) {
	return new StackNode(EXPR, constructNodeOperator(operate, s1, s2));
}

StackNode* construct_Negator_Stack_Node( StackNode* s1) {
	return new StackNode(EXPR, constructNodeNegator(s1));
}

StackNode* reduce(stack<StackNode*> Stack) {	
	if (Stack.size() == 3 && (Stack.top()->type == TERM && Stack.top()->term.token_type == LPAREN_TOKEN)) { //expr --> ( expr )
		Stack.pop();
		StackNode* ex = Stack.top();
		Stack.pop();
		if (Stack.top()->type == TERM && Stack.top()->term.token_type == RPAREN_TOKEN) {
			return ex;
		}
		else {
			syntax_error();
		}
	}
	else if (Stack.size() == 3) {//expr -> expr AND expr || expr OR expr
		StackNode* s1 = Stack.top();
		Stack.pop();
		StackNode* operate = Stack.top();
		Stack.pop();
		StackNode* s2 = Stack.top();
		Stack.pop();
		return construct_Operator_Stack_Node(operate, s1, s2);
	}
	else if(Stack.size() == 2 && (Stack.top()->type == TERM && Stack.top()->term.token_type == NOT_TOKEN)) {//expr -> NOT expr
		Stack.pop();
		StackNode* ex = Stack.top();
        return construct_Negator_Stack_Node(ex);
	}
	else if (Stack.size() == 1) {
		return TERM_TO_EXP(Stack.top());
	}
	else {
		syntax_error();
		return new StackNode();
	}
}

StackNode* Parser::terminalPeek(vector<StackNode*> S){
	if(S.at(S.size()-1)->type==TERM){
		return S.at(S.size()-1);
	}else if(S.at(S.size()-2)->type==TERM){
		return S.at(S.size()-2);
	}else{
		syntax_error();
		return new StackNode();
	}
}


BoolExpression* Parser::parse_sop() {
	buildOperatorPrecedenceTable();
	vector<StackNode*> Stack;	
	Token start;
	Stack.push_back(construct_TERM_Stack_Node(start));
	while (true) {
		Token t1 = lexer.peek(1);
		Token t2 = terminalPeek(Stack)->term;
		
		if (precedanceTable[hash(t2)][hash(t1)]==lessThan || precedanceTable[hash(t2)][hash(t1)] == equall) {
			Token t3 = lexer.GetToken();
			Stack.push_back(construct_TERM_Stack_Node(t3));
		}
		else if(precedanceTable[hash(t2)][hash(t1)] == greaterThan){
			stack<StackNode*> RHS;
			StackNode* last_popped_terminal = terminalPeek(Stack);
			
	
			while (!(Stack.at(Stack.size() - 1)->type == TERM && precedanceTable[hash((Stack.at(Stack.size() - 1))->term)][hash(last_popped_terminal->term)]==lessThan)) {
				
				StackNode* s = Stack.at(Stack.size() - 1);
				
				Stack.pop_back();
				
				if (s->type == TERM) {
					last_popped_terminal = s;
				}
				RHS.push(s);
			} 

			Stack.push_back(reduce(RHS));
		}
		else if (precedanceTable[hash(t2)][hash(t1)] == accept) {
			return Stack.at(1)->expr;
		}
		else {
			syntax_error();
		}
	}	
}

