#include "parser.h"
#include "parser.cpp"
#include "lexer.h"
#include "lexer.cpp"

using namespace std;

// Function to print BooolExpression tree in 2D inorder traversal
//Inspired from https://www.geeksforgeeks.org/print-binary-tree-2-dimensions/
string toStringBoolExpression(BoolExpression* root){
    if (root == NULL)
        return "";
     // Increase distance between levels

	string s = "";
 
	// Process left child
    if(root->type!=VARIABLE){
		s+= toStringBoolExpression(root->left);
	}

	if (root->type == VARIABLE) {
		if(root->SingleCond.is_negated){
			s+= "~";
		}
		s+= "c" + to_string(root->SingleCond.id);
	}
	else if (root->type == OR) {
		s += " + ";
	}else if (root->type == AND) {
		s += " . ";
	}
	else if (root->type == NOT) {
		s += " ~";
	}

	// Process right child first
	if(root->type!=VARIABLE){
		s+= toStringBoolExpression(root->right);
	}
	return s;
}

BoolExpression*  propagarteNegation(BoolExpression* root, bool negated){
	if(root->type==NOT){
		return propagarteNegation(root->right, !negated);
	}else{
		if(negated){
			if(root->type==AND){
				root->type==OR;
			}else if(root->type==OR){
				root->type==AND;
			}else if(root->type==VARIABLE){
				root->SingleCond.is_negated = !root->SingleCond.is_negated;
			}
		}
		if(root->type!=VARIABLE){
			root->left = propagarteNegation(root->left, negated);
			root->right = propagarteNegation(root->right, negated);
		}
		return root;
	}
}
 
// Convert String to BoolExpression
// parses an expression such as c1. c2.c1+~c3 . c1 + c4. ~c1 and stores it in the tree structure
BoolExpression * parse_sop(string str_sop){
	Parser parser(str_sop);
	return propagarteNegation(parser.parse_sop(), false);
} 

// Convert BoolExpression to String
string sop_to_string(BoolExpression* exp1){
	return  toStringBoolExpression(exp1);
}

// AND two expressions
BoolExpression* Bool_And(BoolExpression* exp1, BoolExpression* exp2){
	return new BoolExpression(AND, exp1, exp2);
}

// OR two expressions 
BoolExpression* Bool_Or(BoolExpression* exp1, BoolExpression* exp2){
	return new BoolExpression(OR, exp1, exp2);
}

// Negate an expression (apply DeMorgan's law)
BoolExpression* Bool_Negate(BoolExpression* exp1){
	return propagarteNegation(exp1, true);
}