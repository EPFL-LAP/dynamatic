#ifndef __PARSER__H__
#define __PARSER__H__


#include "lexer.h"


enum expressionType {VARIABLE, OR, AND, NOT, END};

struct BoolExpression{
    expressionType type;

    struct{
        int id;
	    bool is_negated;
    } SingleCond;
    
    
    BoolExpression * left;
    BoolExpression * right;

    BoolExpression(expressionType END) : type(END), SingleCond{-1, false}, left(nullptr), right(nullptr) {}

    //Constructor for single condition nodes
    BoolExpression(expressionType t, int i, bool negated = false) : type(t),  SingleCond{i, negated}, left(nullptr), right(nullptr) {}

    //Constructor for operator nodes
    BoolExpression(expressionType t, BoolExpression* l, BoolExpression* r): type(t), SingleCond{-1, false}, left(l), right(r) {}

};

enum snodeType{EXPR, TERM};

struct StackNode{
    snodeType type;
    union{
        BoolExpression* expr;
        Token term;
    };

    StackNode(): type(EXPR), expr(NULL){};

	StackNode(snodeType t, Token tt) : type(t), term(tt) {};

	StackNode(snodeType t, BoolExpression* e) : type(t), expr(e) {};
};

class Parser {
public:
	LexicalAnalyzer lexer;
    BoolExpression * parse_sop();

    //Constructor
    Parser(string s): lexer(s){};

private:
    int hash(Token t);	
    StackNode* terminalPeek(vector<StackNode*> S);
};

#define COUNT 10


// Function to print BooolExpression tree in 2D inorder traversal
//Inspired from https://www.geeksforgeeks.org/print-binary-tree-2-dimensions/
void printBoolExpression(BoolExpression* root, int space){
    if (root == NULL)
        return;
     // Increase distance between levels
    space += COUNT;
 
    // Process right child first
	if(root->type==OR || root->type==AND||root->type==NOT){
		printBoolExpression(root->right, space);
	}
 
    // Print current node after space
    // count
    cout << endl;
    for (int i = COUNT; i < space; i++)
        cout << " ";

	if (root->type == VARIABLE) {
		cout << "c" << root->SingleCond.id<< "\n";
	}
	else if (root->type == OR) {
		cout << "+ "<< "\n";
	}
	else if (root->type == AND) {
		cout << ". "<< "\n";
	}
	else if (root->type == NOT) {
		cout << "~ "<< "\n";
	}
			
    // Process left child
    if(root->type==OR || root->type==AND){
		printBoolExpression(root->left, space);
	}
}
 
// Wrapper over printBoolExpressionl()
void printBoolExpressionTree(BoolExpression* root){
    // Pass initial space count as 0
    printBoolExpression(root, 0);
}

#endif