//===- BooleanLogicLibrary.h - // Boolean Logic Expression Library Interface
//-----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the interface for the library that handles boolean
// logic expressions.
// It includes functions for propagating negation through a tree using
// DeMorgan's law, and creating boolean expressions. Additionally, it provides
// functions to interact with the Espresso logic minimizer tool. The library
// supports basic operations on boolean expressions such as AND, OR, and
// negation.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_BOOLEANLOGICLIBRARY_H
#define DYNAMATIC_SUPPORT_BOOLEANLOGICLIBRARY_H

#include "dynamatic/Support/BooleanExpression.h"

#include <string>

namespace dynamatic {

// Recursive function that propagates a not operator in a BoolExpression tree by
// applying DeMorgan's law
BoolExpression *propagateNegation(BoolExpression *root, bool negated);

//--------------Espresso--------------

// function to run espresso logic minimzer on a BoolExpression
std::string runEspresso(BoolExpression *expr);

// Convert String to BoolExpression
// parses an expression such as c1. c2.c1+~c3 . c1 + c4. ~c1 and stores it in
// the tree structure
BoolExpression *parseSop(std::string strSop);

// Convert BoolExpression to String
std::string sopToString(BoolExpression *exp1);

// Create an expression of a single variable
BoolExpression *boolVar(std::string id);

// AND two expressions
BoolExpression *boolAnd(BoolExpression *exp1, BoolExpression *exp2);

// OR two expressions
BoolExpression *boolOr(BoolExpression *exp1, BoolExpression *exp2);

// Negate an expression (apply DeMorgan's law)
BoolExpression *boolNegate(BoolExpression *exp1);

// Minimize an expression based on the espresso logic minimizer algorithm
BoolExpression *boolMinimize(BoolExpression *expr);

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_BOOLEANLOGICLIBRARY_H
