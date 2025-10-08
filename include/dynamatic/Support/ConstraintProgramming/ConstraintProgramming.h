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
// Namespace for constraint programming
namespace cp {

/// A single variable in constraint programming
/// Example:
/// auto x = Var("x", Var::INTEGER);
///
/// For simplicity, the upper and lower bounds are not encoded here (just the
/// data types).
struct Var {
  enum VarType { REAL, INTEGER, BOOLEAN };
  std::string name;
  VarType type;
  // Null value of lowerBound would be -inf
  std::optional<double> lowerBound;
  // Null value of upperBound would be +inf
  std::optional<double> upperBound;
  // Using Var as a key
  bool operator<(const Var &other) const noexcept { return name < other.name; }
  bool operator==(const Var &other) const noexcept {
    return name == other.name;
  }

  Var() = default;

  // Explicit constructor:
  // Var newVar = solver.addVariable("newVar", Var::INTEGER, std::nullopt,
  // std::nullopt);
  //
  // Explicit constructor avoids implicit cast from string to Var.
  explicit Var(std::string name, VarType type, std::optional<double> lowerBound,
               std::optional<double> upperBound)
      : name(std::move(name)), type(type), lowerBound(lowerBound),
        upperBound(upperBound) {}
};

/// A linear expression (not an equality/inequality)
/// Example:
///
/// auto x = Var("x", Var::INTEGER);
/// auto y = Var("y", Var::INTEGER);
/// Using operator overloading to construct an expression:
/// auto expr = (x + 2 * y);
struct LinExpr {

  // The coefficients in the linear expression
  // For instance, for x + 2 * y + 1
  // We have (x, 1) and (y, 2)
  std::map<Var, double> terms;
  double constant = 0.0;
  LinExpr() = default;
  LinExpr(const Var &v) { terms[v] = 1.0; }
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

inline LinExpr operator+(const LinExpr &left, const LinExpr &right) {
  LinExpr newExpr = left;
  for (auto &[var, coeff] : right.terms) {
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
inline LinExpr operator*(double c, const Var &v) {
  LinExpr newExpr(v);
  newExpr.terms[v] *= c;
  newExpr.constant *= c;
  return newExpr;
}

/// Overloading mul (commutativity of mul):
/// var * const
inline LinExpr operator*(const Var &v, double c) { return c * v; }

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
  std::map<std::pair<Var, Var>, double> quadTerms;
  QuadExpr() = default;
  QuadExpr(double value) { linexpr = LinExpr(value); }
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
struct LinConstr {
  // The expression
  LinExpr expr;
  Predicate pred;
};

inline LinConstr operator<=(const LinExpr &lhs, const LinExpr &rhs) {
  LinConstr c;
  c.expr = lhs - rhs;
  c.pred = LE;
  return c;
}

inline LinConstr operator>=(const LinExpr &lhs, const LinExpr &rhs) {
  return (rhs <= lhs);
}

inline LinConstr operator==(const LinExpr &lhs, double rhs) {
  LinConstr c;
  c.expr = lhs - rhs;
  c.pred = EQ;
  return c;
}

inline LinConstr operator==(const LinExpr &lhs, const LinExpr &rhs) {
  return (lhs - rhs == 0);
}

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
  virtual ~CPSolver() = default;
  // Virtual class methods: they provide a unified interface for all available
  // solvers.
  virtual Var addVariable(const Var &var) = 0;
  // Create var, add gurobi var, and then return the created variable
  virtual Var addVariable(const std::string &name, Var::VarType type,
                          std::optional<double> lb,
                          std::optional<double> ub) = 0;
  virtual void addLinearConstraint(const LinConstr &constraint,
                                   llvm::StringRef constrName) = 0;
  void addLinearConstraint(const LinConstr &constraint) {
    addLinearConstraint(constraint, "");
  }
  virtual void addQuadConstraint(const QuadConstr &constraint,
                                 llvm::StringRef constrName) = 0;
  virtual void setMaximizeObjective(const LinExpr &expr) = 0;
  virtual void optimize() = 0;
  virtual double getValue(const Var &var) const = 0;
  virtual double getObjective() const = 0;
};

enum MILPSolver {
  COIN_OR_CBC,
#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
  GUROBI,
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
};

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
class GurobiSolver : public CPSolver {

  std::unique_ptr<GRBEnv> env;
  std::map<Var, GRBVar> variables;
  std::unique_ptr<GRBModel> model;

  // Track the added names: prevent adding variables with duplicated names
  std::set<std::string> names;

public:
  GurobiSolver() {
    env = std::make_unique<GRBEnv>(true);
    env->set(GRB_IntParam_OutputFlag, 0);
    env->start();
    model = std::make_unique<GRBModel>(*env);
  }

  Var addVariable(const Var &var) override {
    if (names.count(var.name)) {
      llvm::report_fatal_error("Adding variable with duplicated names is not "
                               "permitted! Aborting...");
    }
    double lb = var.lowerBound.value_or(-GRB_INFINITY);
    double ub = var.upperBound.value_or(GRB_INFINITY);
    char type;
    switch (var.type) {
    case Var::REAL:
      type = GRB_CONTINUOUS;
      break;
    case Var::INTEGER:
      type = GRB_INTEGER;
      break;
    case Var::BOOLEAN:
      type = GRB_BINARY;
    }
    variables[var] = model->addVar(lb, ub, 0.0, type, var.name);
    names.insert(var.name);
    return var;
  }

  // Create var, add gurobi var, and then return
  Var addVariable(const std::string &name, Var::VarType type,
                  std::optional<double> lb, std::optional<double> ub) override {
    auto var = Var(name, type, lb, ub);
    return addVariable(var);
  }

  void addLinearConstraint(const LinConstr &constraint,
                           llvm::StringRef constrName) override {
    GRBLinExpr expr = 0;
    for (auto &[var, coeff] : constraint.expr.terms) {
      expr += coeff * variables[var];
    }
    expr += constraint.expr.constant;
    if (constraint.pred == LE)
      model->addConstr(expr <= 0, constrName.str());
    else if (constraint.pred == EQ)
      model->addConstr(expr == 0, constrName.str());
    else
      llvm_unreachable("Unknown predicate!");
  }

  void addQuadConstraint(const QuadConstr &constraint,
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
      llvm::errs() << "Adding vairable " << name.name << " with coeff " << coeff
                   << " to grb!\n";
      obj += coeff * variables[name];
    }
    obj += expr.constant;
    // NOTE: the constant term can be ignored in the objective
    model->setObjective(obj, GRB_MAXIMIZE);
  }

  void optimize() override {
    model->write("/tmp/model.lp");
    model->optimize();
    model->write("/tmp/model.json");

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

  /// Retrieve the value from the solved MILP
  ///
  /// Example:
  /// auto resultMyVar = solver.getValue(myVar);
  double getValue(const Var &var) const override {
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
};
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

class CbcSolver : public CPSolver {

  OsiClpSolverInterface solver;
  std::map<Var, int> variables; // map Var -> column index
  std::set<std::string> names;

public:
  CbcSolver() {
    // Suppress the solver's output
    solver.messageHandler()->setLogLevel(0);
  }

  Var addVariable(const Var &var) override {
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
    if (var.type == Var::INTEGER)
      solver.setInteger(colIndex);
    else if (var.type == Var::BOOLEAN) {
      solver.setInteger(colIndex);
      solver.setColUpper(colIndex, 1.0);
      solver.setColLower(colIndex, 0.0);
    } // REAL is default

    names.insert(var.name);
    return var;
  }

  Var addVariable(const std::string &name, Var::VarType type,
                  std::optional<double> lb, std::optional<double> ub) override {
    auto var = Var(name, type, lb, ub);
    return addVariable(var);
  }

  void addLinearConstraint(const LinConstr &constraint,
                           llvm::StringRef constrName) override {
    int numCoeffs = constraint.expr.terms.size();
    std::vector<int> indices;
    std::vector<double> values;

    for (auto &[var, coeff] : constraint.expr.terms) {
      indices.push_back(variables.at(var));
      values.push_back(coeff);
    }

    double rowLower, rowUpper;
    if (constraint.pred == LE) {
      rowLower = -1e20;
      rowUpper = -constraint.expr.constant;
    } else if (constraint.pred == EQ) {
      rowLower = -constraint.expr.constant;
      rowUpper = -constraint.expr.constant;
    } else {
      llvm_unreachable("Unknown predicate!");
    }

    CoinPackedVector row;

    // Add elements to the row vector
    row.setVector(numCoeffs, indices.data(), values.data());

    solver.addRow(row, rowLower, rowUpper, constrName.str());
  }

  void addQuadConstraint(const QuadConstr &constraint,
                         llvm::StringRef name) override {
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
    model.setLogLevel(0);

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

  double getValue(const Var &var) const override {
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
};

} // namespace cp
} // namespace dynamatic
