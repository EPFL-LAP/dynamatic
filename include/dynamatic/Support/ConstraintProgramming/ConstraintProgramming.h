/// ConstraintProgramming.h
/// This header defines a DSL for constraint programming
/// - Users can use overloaded '+', '*', '-', '<', etc. to construct constraints
/// and objectives.
/// - It is designed to be solver agnostic.
#pragma once
#include "llvm/Support/ErrorHandling.h"
#include <map>
#include <memory>
#include <optional>
#include <string>

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
  // Null value of upperBound would be -inf
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
};

inline LinearExpr operator+(const LinearExpr &left, const LinearExpr &right) {
  LinearExpr r = left;
  for (auto &[var, coefficients] : right.coefficients) {
    if (r.coefficients.count(var))
      r.coefficients[var] += coefficients;
    else
      r.coefficients[var] = coefficients;
  }
  r.constant += right.constant;
  return r;
}

inline LinearExpr operator-(const LinearExpr &left, const LinearExpr &right) {
  LinearExpr r = left;
  for (auto &[var, coefficients] : right.coefficients) {
    if (r.coefficients.count(var))
      r.coefficients[var] -= coefficients;
    else
      r.coefficients[var] = -coefficients;
  }
  r.constant -= right.constant;
  return r;
}

/// const * var
inline LinearExpr operator*(double c, const LinearExpr &expr) {
  LinearExpr r;
  for (auto &[v, coefficients] : expr.coefficients)
    r.coefficients[v] = c * coefficients;
  r.constant = c * expr.constant;
  return r;
}

/// Commutativity of mul
/// var * const
inline LinearExpr operator*(const LinearExpr &a, double c) { return c * a; }

/// Adding var and constant:
/// var + const
inline LinearExpr operator+(const Var &v, double c) {
  LinearExpr e(v);
  e.constant += c;
  return e;
}

/// Overloading add (commutativity of add):
/// const + var
inline LinearExpr operator+(double c, const Var &v) { return v + c; }

/// Overloading mul
/// const * var
inline LinearExpr operator*(double c, const Var &v) {
  LinearExpr e(v);
  e.coefficients[v] *= c;
  return e;
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
  virtual ~CPSolver() = default;

  virtual Var addVariable(const Var &var) = 0;
  virtual void addLinearConstraint(const Constraint &constraint) = 0;
  virtual void setMaximizeObjective(const LinearExpr &expr) = 0;
  virtual bool solve() = 0;
  virtual double getValue(const Var &var) const = 0;
};

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
class GurobiSolver : CPSolver {

  std::unique_ptr<GRBEnv> env;
  std::unique_ptr<GRBModel> model;
  std::map<Var, GRBVar> variables;

public:
  GurobiSolver() {
    env = std::make_unique<GRBEnv>(true);
    env->start();
    model = std::make_unique<GRBModel>(*env);
  }

  Var addVariable(const Var &var) override {
    double lb =
        var.lowerBound.has_value() ? var.lowerBound.value() : -GRB_INFINITY;
    double ub =
        var.upperBound.has_value() ? var.upperBound.value() : GRB_INFINITY;
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
    return var;
  }

  // Create var, add gurobi var, and then return
  Var addVariable(const std::string &name, Var::VarType type,
                  std::optional<double> lb, std::optional<double> ub) {
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
    // NOTE: the constant term can be ignored in the objective
    model->setObjective(obj, GRB_MAXIMIZE);
  }

  bool solve() override {
    model->optimize();
    return model->get(GRB_IntAttr_Status) == GRB_OPTIMAL;
  }

  /// Retrieve the value from the solved MILP
  ///
  /// Example:
  /// auto resultMyVar = solver.getValue(myVar);
  double getValue(const Var &var) const override {
    return variables.at(var).get(GRB_DoubleAttr_X);
  }
};
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

} // namespace cp
} // namespace dynamatic
