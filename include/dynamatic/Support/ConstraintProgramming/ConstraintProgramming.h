/// ConstraintProgramming.h
/// This header defines a DSL for constraint programming
/// - Users can use overloaded '+', '*', '-', '<', etc. to construct constraints
/// and objectives.
/// - It is designed to be solver agnostic.
///
/// For example usage of this API, please refer to
/// `dynamatic/unittests/Support/ConstraintProgramming/CPTest.cpp`
#pragma once
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "coin/CbcModel.hpp"
#include "coin/OsiClpSolverInterface.hpp"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

namespace dynamatic {

/// A single variable in constraint programming
/// Example:
/// A linear expression (not an equality/inequality)
/// Examples:
///
/// Creating variables:
/// 1. An integer variable without upper and lower bounds
/// auto x = Var("x", Var::INTEGER);
/// 2. An float variable without lower bound and has a upperbound of 1
/// auto y = Var("y", Var::REAL, std::nullopt, 1);
struct CPVar {
  enum VarType { REAL, INTEGER, BOOLEAN };
  std::string name;
  VarType type;
  // Null value of lowerBound would be -inf
  std::optional<double> lowerBound;
  // Null value of upperBound would be +inf
  std::optional<double> upperBound;
  // Using Var as a key
  bool operator<(const CPVar &other) const noexcept {
    return name < other.name;
  }

  CPVar() = default;

  // Explicit constructor:
  // Var newVar = solver.addVariable("newVar", Var::INTEGER, std::nullopt,
  // std::nullopt);
  //
  // Explicit constructor avoids implicit cast from string to Var.
  explicit CPVar(std::string name, VarType type,
                 std::optional<double> lowerBound = /* -inf */ std::nullopt,
                 std::optional<double> upperBound = /* +inf */ std::nullopt)
      : name(std::move(name)), type(type), lowerBound(lowerBound),
        upperBound(upperBound) {}
};

/// An expression in constraint programming.
///
/// Examples:
/// 1. Using operator overloading to construct an expression:
/// auto expr1 = (x + 2 * y);
/// 2. Constructing constraints from 2 expressions
/// auto expr2 = (3 * z);
/// Constraint constr1 = (expr1 <= expr2);
struct LinExpr {

  // The coefficients in the linear expression
  // For instance, for x + 2 * y + 1
  // We have (x, 1) and (y, 2)
  std::map<CPVar, double> terms;
  double constant = 0.0;
  LinExpr() = default;
  LinExpr(const CPVar &v) { terms[v] = 1.0; }
  LinExpr(double value) { constant = value; }
  LinExpr operator-() const {
    LinExpr negated;
    for (auto &[var, coeff] : terms) {
      negated.terms[var] = -coeff;
    }
    negated.constant = -constant;
    return negated;
  }
};

inline LinExpr operator+(const CPVar &left, double right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  // REMARK:
  // std::map makes this safe when var does not exist in "right" at the
  // first place by calling the default constructor, i.e., setting terms[right]
  // = 0.0.
  newExpr.constant = right;
  return newExpr;
}

inline LinExpr operator+(double left, const CPVar &right) {
  return (right + left);
}

inline LinExpr operator-(const CPVar &left, double right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  // REMARK:
  // std::map makes this safe when var does not exist in "right" at the
  // first place by calling the default constructor, i.e., setting terms[right]
  // = 0.0.
  newExpr.constant = -right;
  return newExpr;
}

inline LinExpr operator-(double left, const CPVar &right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = right;
  newExpr.terms[right] = -1.0;
  // REMARK:
  // std::map makes this safe when var does not exist in "right" at the
  // first place by calling the default constructor, i.e., setting terms[right]
  // = 0.0.
  newExpr.constant = left;
  return newExpr;
}

inline LinExpr operator+(const CPVar &left, const CPVar &right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  // REMARK:
  // std::map makes this safe when var does not exist in "right" at the
  // first place by calling the default constructor, i.e., setting terms[right]
  // = 0.0.
  newExpr.terms[right] += 1.0;
  return newExpr;
}

inline LinExpr operator-(const CPVar &left, const CPVar &right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  // REMARK:
  // std::map makes this safe when var does not exist in "right" at the
  // first place by calling the default constructor, i.e., setting terms[right]
  // = 0.0.
  newExpr.terms[right] -= 1.0;
  return newExpr;
}

inline LinExpr operator+(const LinExpr &left, double right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  newExpr.constant += right;
  return newExpr;
}

inline LinExpr operator+(double left, const LinExpr &right) {
  // REMARK:
  // this is a deep copy of "left"
  return (right + left);
}

inline LinExpr operator+(const LinExpr &left, const LinExpr &right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  for (auto &[var, coeff] : right.terms) {
    // REMARK:
    // std::map makes this safe when var does not exist in "left" at the
    // first place by calling the default constructor, i.e., setting terms[var]
    // = 0.0. This happens when "left" and "right" have different variables.
    newExpr.terms[var] += coeff;
  }
  newExpr.constant += right.constant;
  return newExpr;
}

inline void operator+=(LinExpr &left, const LinExpr &right) {
  for (auto &[var, coeff] : right.terms) {
    left.terms[var] += coeff;
  }
  left.constant += right.constant;
}

inline LinExpr operator-(const LinExpr &left, const LinExpr &right) {
  LinExpr newExpr = left;
  for (auto &[var, coeff] : right.terms) {
    newExpr.terms[var] -= coeff;
  }
  newExpr.constant -= right.constant;
  return newExpr;
}

inline void operator-=(LinExpr &left, const LinExpr &right) {
  for (auto &[var, coeff] : right.terms) {
    left.terms[var] -= coeff;
  }
  left.constant -= right.constant;
}

/// Overloading mul
/// const * var
inline LinExpr operator*(double c, const CPVar &v) {
  LinExpr newExpr(v);
  newExpr.terms[v] *= c;
  newExpr.constant *= c;
  return newExpr;
}

/// Overloading mul (commutativity of mul):
/// var * const
inline LinExpr operator*(const CPVar &v, double c) { return c * v; }

/// Overloading mul
/// const * linexpr
inline LinExpr operator*(double c, const LinExpr &expr) {
  LinExpr newExpr(expr);
  for (auto &[var, coeff] : newExpr.terms) {
    newExpr.terms[var] = c * coeff;
  }
  newExpr.constant *= c;
  return newExpr;
}

inline LinExpr operator*(const LinExpr &expr, double c) { return c * expr; }

struct QuadExpr {
  LinExpr linexpr;
  std::map<std::pair<CPVar, CPVar>, double> quadTerms;
  QuadExpr() = default;
  QuadExpr(double value) { linexpr = LinExpr(value); }
  QuadExpr(const CPVar &var) { linexpr = LinExpr(var); }
  QuadExpr(const LinExpr &expr) { linexpr = expr; }
};

inline QuadExpr operator*(const LinExpr &lhs, const LinExpr &rhs) {
  QuadExpr e;
  // Quadratic terms:
  for (auto &[lhsTerm, lhsCoeff] : lhs.terms) {
    for (auto &[rhsTerm, rhsCoeff] : rhs.terms) {
      e.quadTerms[std::make_pair(lhsTerm, rhsTerm)] = lhsCoeff * rhsCoeff;
    }
  }
  // Linear terms:
  for (auto &[v, coeff] : lhs.terms) {
    e.linexpr.terms[v] += coeff * rhs.constant;
  }
  for (auto &[v, coeff] : rhs.terms) {
    e.linexpr.terms[v] += coeff * lhs.constant;
  }
  // Constant term:
  e.linexpr.constant = lhs.constant * rhs.constant;
  return e;
}

inline QuadExpr operator*(double lhs, const QuadExpr &rhs) {
  QuadExpr e = rhs;
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : e.quadTerms) {
    e.quadTerms[quadTerm] = coeff * lhs;
  }
  for (auto &[linTerm, coeff] : e.linexpr.terms) {
    e.linexpr.terms[linTerm] = coeff * lhs;
  }
  return e;
}

inline QuadExpr operator+(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = lhs;
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    e.quadTerms[quadTerm] += coeff;
  }
  // Linear and constant terms:
  e.linexpr = e.linexpr + rhs.linexpr;
  return e;
}

inline QuadExpr operator-(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = lhs;
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    e.quadTerms[quadTerm] -= coeff;
  }
  // Linear and constant terms:
  e.linexpr = e.linexpr - rhs.linexpr;
  return e;
}

inline void operator+=(QuadExpr lhs, const QuadExpr &rhs) {
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    lhs.quadTerms[quadTerm] += coeff;
  }
  // Linear and constant terms:
  lhs.linexpr = lhs.linexpr + rhs.linexpr;
}

inline void operator-=(QuadExpr lhs, const QuadExpr &rhs) {
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    lhs.quadTerms[quadTerm] -= coeff;
  }
  // Linear and constant terms:
  lhs.linexpr = lhs.linexpr - rhs.linexpr;
}

enum Predicate {
  /* <= */ LE,
  /* == */ EQ
};

/// Class for constraints
/// It has the form:
/// [expr] [pred] 0
/// For example:
/// - x + 2 * y - z - 1 <= 1
/// - x + 2 * y - z + 1 <= 0
/// - x + 2 * y - z + 2 == 0
/// The rhs is always 0

struct QuadConstr {
  // The expression
  QuadExpr expr;
  Predicate pred;
};

inline QuadConstr operator<=(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = (lhs - rhs);
  QuadConstr c;
  c.expr = e;
  c.pred = LE;
  return c;
}

inline QuadConstr operator>=(const QuadExpr &lhs, const QuadExpr &rhs) {
  return (rhs <= lhs);
}

inline QuadConstr operator==(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = (lhs - rhs);
  QuadConstr c;
  c.expr = e;
  c.pred = EQ;
  return c;
}

/// Abstract base class for different solvers (e.g., Gurobi, Google's solvers,
/// or Cbc).
///
/// This is overloaded for the Gurobi solver and the Google's OR Tools API.
class CPSolver {
public:
  enum Status { OPTIMAL, NONOPTIMAL, INFEASIBLE, UNBOUNDED, UNKNOWN, ERROR };
  Status status = UNKNOWN;

  // LLVM Implementation of rtti functions like dyn_cast<>, isa<> needs these
  // function
  // [START LLVM RTTI prerequisites]
  enum SolverKind {
#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
    GUROBI,
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
    CBC,
  };
  SolverKind solverKind;
  SolverKind getKind() const { return solverKind; }
  static inline bool classof(CPSolver const *) { return true; }
  // [END LLVM RTTI prerequisites]

  // Solver timeout in second.
  // If timeout <= 0, then this option is ignored.
  int timeout;
  CPSolver(int t, SolverKind k) : solverKind(k), timeout(t) {}

  virtual ~CPSolver() = default;
  // Virtual class methods: they provide a unified interface for all available
  // solvers.
  virtual CPVar addVar(const CPVar &var) = 0;
  // Create var, add gurobi var, and then return the created variable
  virtual CPVar addVar(const std::string &name, CPVar::VarType type,
                       std::optional<double> lb, std::optional<double> ub) = 0;
  virtual void addConstr(const QuadConstr &constraint,
                         llvm::StringRef constrName) = 0;
  void addConstr(const QuadConstr &constraint) { addConstr(constraint, ""); }
  virtual void addQConstr(const QuadConstr &constraint,
                          llvm::StringRef constrName) = 0;
  virtual void setMaximizeObjective(const LinExpr &expr) = 0;
  virtual void optimize() = 0;
  virtual double getValue(const CPVar &var) const = 0;
  virtual double getObjective() const = 0;

  virtual void write(llvm::StringRef filePath) const = 0;

  virtual void writeSol(llvm::StringRef filePath) const = 0;
};

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
class GurobiSolver : public CPSolver {

  std::unique_ptr<GRBEnv> env;
  std::map<CPVar, GRBVar> variables;
  std::unique_ptr<GRBModel> model;

  // Track the added names: prevent adding variables with duplicated names
  std::set<std::string> names;

public:
  GurobiSolver(int timeout = -1 /* default = no timeout*/)
      : CPSolver(timeout, GUROBI) {
    env = std::make_unique<GRBEnv>(true);
    env->set(GRB_IntParam_OutputFlag, 0);
    env->start();

    if (timeout > 0) {
      env->set(GRB_DoubleParam_TimeLimit, timeout);
    }

    model = std::make_unique<GRBModel>(*env);
  }

  CPVar addVar(const CPVar &var) override {
    if (names.count(var.name)) {
      llvm::report_fatal_error("Adding variable with duplicated names is not "
                               "permitted! Aborting...");
    }
    double lb = var.lowerBound.value_or(-GRB_INFINITY);
    double ub = var.upperBound.value_or(GRB_INFINITY);
    char type;
    switch (var.type) {
    case CPVar::REAL:
      type = GRB_CONTINUOUS;
      break;
    case CPVar::INTEGER:
      type = GRB_INTEGER;
      break;
    case CPVar::BOOLEAN:
      type = GRB_BINARY;
    }
    variables[var] = model->addVar(lb, ub, 0.0, type, var.name);
    names.insert(var.name);
    return var;
  }

  // Create var, add gurobi var, and then return
  CPVar addVar(const std::string &name, CPVar::VarType type,
               std::optional<double> lb, std::optional<double> ub) override {
    auto var = CPVar(name, type, lb, ub);
    return addVar(var);
  }

  void addConstr(const QuadConstr &constraint,
                 llvm::StringRef constrName) override {
    if (!constraint.expr.quadTerms.empty())
      llvm::report_fatal_error(
          "Adding a linear constraint with quadratic terms!");
    addQConstr(constraint, constrName);
  }

  void addQConstr(const QuadConstr &constraint,
                  llvm::StringRef constrName) override {
    GRBQuadExpr expr = 0;

    // Quadratic terms
    for (auto &[name, coeff] : constraint.expr.quadTerms) {
      expr += coeff * variables[name.first] * variables[name.second];
    }

    // Linear terms
    for (auto &[name, coeff] : constraint.expr.linexpr.terms) {
      expr += coeff * variables[name];
    }

    // Constant terms
    expr += constraint.expr.linexpr.constant;
    if (constraint.pred == LE)
      model->addQConstr(expr <= 0, constrName.str());
    else if (constraint.pred == EQ)
      model->addQConstr(expr == 0, constrName.str());
    else
      llvm_unreachable("Unknown predicate!");
  }

  void setMaximizeObjective(const LinExpr &expr) override {
    GRBLinExpr obj = 0;
    for (auto &[name, coeff] : expr.terms) {
      obj += coeff * variables[name];
    }
    obj += expr.constant;
    // NOTE: the constant term can be ignored in the objective
    model->setObjective(obj, GRB_MAXIMIZE);
  }

  void optimize() override {
    model->optimize();

    switch (model->get(GRB_IntAttr_Status)) {
    case GRB_OPTIMAL:
      status = OPTIMAL;
      break;
    case GRB_SUBOPTIMAL:
    case GRB_TIME_LIMIT:
      status = NONOPTIMAL;
      break;
    case GRB_UNBOUNDED:
      status = UNBOUNDED;
      break;
    case GRB_INFEASIBLE:
      status = INFEASIBLE;
      break;
    default:
      status = ERROR;
    }
  }

  void write(llvm::StringRef filePath) const override {
    model->write(filePath.str());
  }

  void writeSol(llvm::StringRef filePath) const override {
    if (status != OPTIMAL && status != NONOPTIMAL) {
      llvm::report_fatal_error("Calling writeSol before the model was solved!");
    }

    std::ofstream myfile(filePath.str());
    if (myfile.is_open()) {
      for (auto &[var, _] : variables) {
        myfile << var.name << " = " << getValue(var) << "\n";
      }
    } else {
      llvm::errs() << "Unable to open file: " << filePath << "!\n";
      llvm::report_fatal_error("Unable to open file!");
    }
  }

  /// Retrieve the value from the solved MILP
  ///
  /// Example:
  /// auto resultMyVar = solver.getValue(myVar);
  double getValue(const CPVar &var) const override {
    if (status != OPTIMAL && status != NONOPTIMAL) {
      llvm::errs() << "Solution is not available while retrieving " << var.name
                   << "!\n";
      llvm::report_fatal_error("Cannot retrieve the value of variable!");
    }
    return variables.at(var).get(GRB_DoubleAttr_X);
  }

  double getObjective() const override {
    if (status != OPTIMAL && status != NONOPTIMAL) {
      llvm::report_fatal_error("Cannot retrieve the objective because the "
                               "solution is not available!");
    }
    return model->get(GRB_DoubleAttr_ObjVal);
  }

  // [START LLVM RTTI prerequisites]
  static bool classof(const CPSolver *b) { return b->getKind() == GUROBI; }
  static bool classof(const GurobiSolver *b) { return true; }
  // [END LLVM RTTI prerequisites]
};
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

class CbcSolver : public CPSolver {

  OsiClpSolverInterface solver;
  std::map<CPVar, int> variables; // map Var -> column index
  std::set<std::string> names;

public:
  CbcSolver(int timeout = -1 /* default = no timeout */)
      : CPSolver(timeout, CBC) {
    // Suppress the solver's output
    //
    //
    solver.messageHandler()->setLogLevel(-1);
    solver.getModelPtr()->messageHandler()->setLogLevel(-1);
  }

  CPVar addVar(const CPVar &var) override {
    if (names.count(var.name)) {
      llvm::report_fatal_error("Adding variable with duplicated names is not "
                               "permitted! Aborting...");
    }

    double lb = var.lowerBound.value_or(-1e20);
    double ub = var.upperBound.value_or(1e20);

    int colIndex = solver.getNumCols();

    // Add an empty column for this variable
    solver.addCol(0, nullptr, nullptr, lb, ub, 0.0);
    variables[var] = colIndex;

    // Set variable type
    if (var.type == CPVar::INTEGER)
      solver.setInteger(colIndex);
    else if (var.type == CPVar::BOOLEAN) {
      solver.setInteger(colIndex);
      solver.setColUpper(colIndex, 1.0);
      solver.setColLower(colIndex, 0.0);
    } // REAL is default

    names.insert(var.name);
    return var;
  }

  CPVar addVar(const std::string &name, CPVar::VarType type,
               std::optional<double> lb, std::optional<double> ub) override {
    auto var = CPVar(name, type, lb, ub);
    return addVar(var);
  }

  void addConstr(const QuadConstr &constraint,
                 llvm::StringRef constrName) override {
    if (!constraint.expr.quadTerms.empty()) {
      llvm::report_fatal_error(
          "Cbc solver does not support quadratic constraints! Aborting");
    }

    auto linearPart = constraint.expr.linexpr;

    int numCoeffs = linearPart.terms.size();
    std::vector<int> indices;
    std::vector<double> values;

    for (auto &[var, coeff] : constraint.expr.linexpr.terms) {
      indices.push_back(variables.at(var));
      values.push_back(coeff);
    }

    double rowLower, rowUpper;
    if (constraint.pred == LE) {
      rowLower = -1e20;
      rowUpper = -linearPart.constant;
    } else if (constraint.pred == EQ) {
      rowLower = -linearPart.constant;
      rowUpper = -linearPart.constant;
    } else {
      llvm_unreachable("Unknown predicate!");
    }

    CoinPackedVector row;

    // Add elements to the row vector
    row.setVector(numCoeffs, indices.data(), values.data());

    solver.addRow(row, rowLower, rowUpper, constrName.str());
  }

  void addQConstr(const QuadConstr &constraint, llvm::StringRef name) override {
    llvm::report_fatal_error(
        "Quadratic constraints is currently unavailable for CBC!");
  }

  void setMaximizeObjective(const LinExpr &expr) override {
    // Create objective array
    std::vector<double> obj(solver.getNumCols(), 0.0);
    for (auto &[var, coeff] : expr.terms) {
      // Set objective in solver
      solver.setObjCoeff(variables.at(var), coeff);
    }

    // CBC minimizes by default; for maximize, multiply by -1
    solver.setObjSense(-1.0);
  }

  void optimize() override {
    CbcModel model(solver);
    // Disable all output
    // 0 = no output, 1 = minimal, higher = more verbose
    model.setLogLevel(-1);

    if (this->timeout > 0) {
      model.setMaximumSeconds(timeout);
    }

    model.branchAndBound();

    int stat = model.status();
    if (stat == 0) // optimal
      status = OPTIMAL;
    else if (stat == 1) // stopped, feasible solution
      status = NONOPTIMAL;
    else if (stat == 2) // infeasible
      status = INFEASIBLE;
    else if (stat == 3) // unbounded
      status = UNBOUNDED;
    else
      status = ERROR;

    // Update solver with solution for getValue
    solver.setColSolution(model.bestSolution());
  }

  void write(llvm::StringRef filePath) const override {
    // HACK: This implementation bypasses the log level and prints some warning
    // messages to stdout; this pollutes the final mlir output.
    //
    // Therefore, this write command currently doesn't do anything.
    //
    // solver.writeLp(filePath.str().c_str());
  }

  void writeSol(llvm::StringRef filePath) const override {
    if (status != OPTIMAL && status != NONOPTIMAL) {
      llvm::report_fatal_error("Calling writeSol before the model was solved!");
    }

    std::ofstream solLogFile(filePath.str());
    if (solLogFile.is_open()) {
      for (auto &[var, _] : variables) {
        solLogFile << var.name << " = " << getValue(var) << "\n";
      }
    } else {
      llvm::errs() << "Unable to open file: " << filePath << "!\n";
      llvm::report_fatal_error("Unable to open file!");
    }
  }

  double getValue(const CPVar &var) const override {
    if (status != OPTIMAL && status != NONOPTIMAL) {
      llvm::errs() << "Solution is not available while retrieving " << var.name
                   << "!\n";
      llvm::report_fatal_error("Cannot retrieve the value of variable!");
    }
    return solver.getColSolution()[variables.at(var)];
  }

  double getObjective() const override {
    if (status != OPTIMAL && status != NONOPTIMAL) {
      llvm::report_fatal_error("Cannot retrieve the objective because the "
                               "solution is not available!");
    }
    return solver.getObjValue();
  }

  // [START LLVM RTTI prerequisites]
  static bool classof(const CbcSolver *b) { return true; }
  static bool classof(const CPSolver *b) { return b->getKind() == CBC; }
  // [END LLVM RTTI prerequisites]
};

} // namespace dynamatic
