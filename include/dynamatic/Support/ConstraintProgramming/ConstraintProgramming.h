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

  // Non-explicit constructor for
  // Var newVar = solver.addVariable({"newVar", Var::INTEGER})
  Var(std::string name, VarType type, std::optional<double> lowerBound,
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
struct LinearExpr {

  // The coefficients in the linear expression
  // For instance, for x + 2 * y + 1
  // We have (x, 1) and (y, 2)
  std::map<Var, double> coefficients;
  double constant = 0.0;
  LinearExpr() = default;
  LinearExpr(const Var &v) { coefficients[v] = 1.0; }
  LinearExpr(double value) { constant = value; }

  LinearExpr operator-() const {
    LinearExpr negated;
    for (auto &[var, coeff] : coefficients) {
      negated.coefficients[var] = -coeff;
    }
    negated.constant = -constant;
    return negated;
  }
};

inline LinearExpr operator+(const LinearExpr &left, const LinearExpr &right) {
  LinearExpr newExpr = left;
  for (auto &[var, coeff] : right.coefficients) {
    if (newExpr.coefficients.count(var))
      newExpr.coefficients[var] += coeff;
    else
      newExpr.coefficients[var] = coeff;
  }
  newExpr.constant += right.constant;
  return newExpr;
}

inline LinearExpr operator-(const LinearExpr &left, const LinearExpr &right) {
  LinearExpr newExpr = left;
  for (auto &[var, coeff] : right.coefficients) {
    if (newExpr.coefficients.count(var))
      newExpr.coefficients[var] -= coeff;
    else
      newExpr.coefficients[var] = -coeff;
  }
  newExpr.constant -= right.constant;
  return newExpr;
}

/// Overloading mul
/// const * var
inline LinearExpr operator*(double c, const Var &v) {
  LinearExpr newExpr(v);
  newExpr.coefficients[v] *= c;
  newExpr.constant *= c;
  return newExpr;
}

/// Overloading mul (commutativity of mul):
/// var * const
inline LinearExpr operator*(const Var &v, double c) { return c * v; }

/// Class for constraints
/// It has the form:
/// [expr] [pred] 0
/// For example:
/// - x + 2 * y - z - 1 <= 1
/// - x + 2 * y - z + 1 <= 0
/// - x + 2 * y - z + 2 == 0
/// The rhs is always 0
struct Constraint {
  enum Predicate {
    /* <= */ LE,
    /* == */ EQ
  };

  // The expression
  LinearExpr expr;
  Predicate pred;
};

inline Constraint operator<=(const LinearExpr &lhs, double rhs) {
  Constraint c;
  for (auto &[v, coeff] : lhs.coefficients)
    c.expr.coefficients[v] = coeff;
  c.expr.constant = lhs.constant - rhs;
  c.pred = Constraint::LE;
  return c;
}

inline Constraint operator<=(const LinearExpr &lhs, const LinearExpr &rhs) {
  return ((lhs - rhs) <= 0.0);
}

inline Constraint operator>=(double lhs, const LinearExpr &rhs) {
  return (rhs <= lhs);
}

inline Constraint operator>=(const LinearExpr &lhs, const LinearExpr &rhs) {
  return (rhs <= lhs);
}

inline Constraint operator>=(const LinearExpr &lhs, double rhs) {
  return (rhs - lhs <= 0);
}

inline Constraint operator==(const LinearExpr &lhs, double rhs) {
  Constraint c;
  for (auto &[v, coeff] : lhs.coefficients)
    c.expr.coefficients[v] = coeff;
  c.expr.constant = lhs.constant - rhs;
  c.pred = Constraint::EQ;
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
  virtual void addLinearConstraint(const Constraint &constraint) = 0;
  virtual void setMaximizeObjective(const LinearExpr &expr) = 0;
  virtual void optimize() = 0;
  virtual std::optional<double> getValue(const Var &var) const = 0;
  virtual std::optional<double> getObjective() const = 0;
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

  void addLinearConstraint(const Constraint &constraint) override {
    GRBLinExpr expr = 0;
    for (auto &[name, coeff] : constraint.expr.coefficients) {
      expr += coeff * variables[name];
    }
    expr += constraint.expr.constant;
    if (constraint.pred == Constraint::LE)
      model->addConstr(expr <= 0);
    else if (constraint.pred == Constraint::EQ)
      model->addConstr(expr == 0);
    else
      llvm_unreachable("Unknown predicate!");
  }

  void setMaximizeObjective(const LinearExpr &expr) override {
    GRBLinExpr obj = 0;
    for (auto &[name, coeff] : expr.coefficients) {
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

  /// Retrieve the value from the solved MILP
  ///
  /// Example:
  /// auto resultMyVar = solver.getValue(myVar);
  std::optional<double> getValue(const Var &var) const override {
    if (status != OPTIMAL && status != NONOPTIMAL)
      return std::nullopt;
    return variables.at(var).get(GRB_DoubleAttr_X);
  }

  std::optional<double> getObjective() const override {
    if (status != OPTIMAL && status != NONOPTIMAL)
      return std::nullopt;
    return model->get(GRB_DoubleAttr_ObjVal);
  }
};
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

class CbcSolver : public CPSolver {

  OsiClpSolverInterface solver;
  std::map<Var, int> variables; // map Var -> column index
  std::set<std::string> names;

public:
  CbcSolver() = default;

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

  void addLinearConstraint(const Constraint &constraint) override {
    int numCoeffs = constraint.expr.coefficients.size();
    std::vector<int> indices;
    std::vector<double> values;

    for (auto &[var, coeff] : constraint.expr.coefficients) {
      indices.push_back(variables.at(var));
      values.push_back(coeff);
    }

    double rowLower, rowUpper;
    if (constraint.pred == Constraint::LE) {
      rowLower = -1e20;
      rowUpper = -constraint.expr.constant;
    } else if (constraint.pred == Constraint::EQ) {
      rowLower = -constraint.expr.constant;
      rowUpper = -constraint.expr.constant;
    } else {
      llvm_unreachable("Unknown predicate!");
    }

    solver.addRow(numCoeffs, indices.data(), values.data(), rowLower, rowUpper);
  }

  void setMaximizeObjective(const LinearExpr &expr) override {
    // Create objective array
    std::vector<double> obj(solver.getNumCols(), 0.0);
    for (auto &[var, coeff] : expr.coefficients) {
      // Set objective in solver
      solver.setObjCoeff(variables.at(var), coeff);
    }

    // CBC minimizes by default; for maximize, multiply by -1
    solver.setObjSense(-1.0);
  }

  void optimize() override {
    CbcModel model(solver);
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

  std::optional<double> getValue(const Var &var) const override {
    if (status != OPTIMAL && status != NONOPTIMAL)
      return std::nullopt;
    return solver.getColSolution()[variables.at(var)];
  }

  std::optional<double> getObjective() const override {
    if (status != OPTIMAL && status != NONOPTIMAL)
      return std::nullopt;
    return solver.getObjValue();
  }
};

} // namespace cp
} // namespace dynamatic
