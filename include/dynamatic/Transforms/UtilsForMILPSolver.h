//===- UtilsForMILPSolver.h - Gurobi solver for MILP  -----------*- C++ -*-===//
//
// This file MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_UTILSFORMILPSOLVER_H
#define DYNAMATIC_TRANSFORMS_UTILSFORMILPSOLVER_H

#include "dynamatic/Support/LLVM.h"

enum VarType {REAL, INTEGER, BOOLEAN};  /// Types of variables
enum RowType {LEQ, GEQ, EQ};            /// Type of constraints
enum Status {OPTIMAL, NONOPTIMAL, UNFEASIBLE, UNBOUNDED, UNKNOWN, ERROR};  

struct Var {
        std::string name;
        VarType type;
        double lower_bound;
        double upper_bound; // unbounded if upper_bound < lower_bound
        double value;
    };

#endif // DYNAMATIC_TRANSFORMS_UTILSFORMILPSOLVER_H