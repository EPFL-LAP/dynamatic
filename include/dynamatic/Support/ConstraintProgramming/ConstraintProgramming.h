/// ConstraintProgramming.h
/// This header defines a DSL for constraint programming
/// - Users can use overloaded '+', '*', '-', '<', etc. to construct constraints
/// and objectives.
/// - It is designed to be solver agnostic.
/// - The API is designed to look very similar to gurobi's API.
///
/// For example usage of this API, please refer to
/// `dynamatic/unittests/Support/ConstraintProgramming/CPTest.cpp`
#pragma once
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

#ifdef DYNAMATIC_ENABLE_CBC
#include "coin/CbcModel.hpp"
#include "coin/OsiClpSolverInterface.hpp"
#endif // DYNAMATIC_ENABLE_CBC

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

namespace dynamatic {

enum VarType { REAL, INTEGER, BOOLEAN };
/// Forward declaration
namespace detail {
/// Remark: the user of "ConstraintProgramming.h" should not use anything in
/// detail::.
///
/// This struct contains the actual storage of CPVar
struct CPVarImpl {
  std::string name;
  VarType type;
  // Null value of lowerBound would be -inf
  std::optional<double> lowerBound;
  // Null value of upperBound would be +inf
  std::optional<double> upperBound;
  // Using Var as a key

  CPVarImpl() = default;
  // Explicit constructor:
  // Var newVar = solver.addVariable("newVar", Var::INTEGER, std::nullopt,
  // std::nullopt);
  //
  // Explicit constructor avoids implicit cast from string to Var.
  explicit CPVarImpl(llvm::StringRef name, VarType type,
                     std::optional<double> lowerBound = /* -inf */ std::nullopt,
                     std::optional<double> upperBound = /* +inf */ std::nullopt)
      : name(name), type(type), lowerBound(lowerBound), upperBound(upperBound) {
  }
};
} // namespace detail

/// A single variable in constraint programming
///
/// Examples:
/// Creating variables:
/// 1. An integer variable without upper and lower bounds
/// auto x = Var("x", Var::INTEGER);
/// 2. An float variable without lower bound and has a upperbound of 1
/// auto y = Var("y", Var::REAL, std::nullopt, 1);
///
/// It is a wrapper class around a pointer
/// reference to the actual storage (CPVarImpl). The rationale behind this is
/// the following:
/// - We want to use DSL-like operation to construct LP constraints and
/// objectives.
/// - Storing the pointer version of CPVar in LinExpr/QuadExpr/CPSolver is very
/// painful, and we cannot overload arithmetic operations easily on them.
/// - Putting the storage directly in CPVar causes a lot of copying, which is a
/// huge overhead when there are many of them.
///
/// For example, the copy assignment
///
/// CPVar a("x", INTEGER);
/// CPVar b("y", INTEGER);
/// b = a; // <--- this one
///
/// Would use the implicitly generated copy constructor to do member wise
/// assignment (this->impl = other.impl;), which only copies the pointer but not
/// the underlying memory.
struct CPVar {
  std::shared_ptr<detail::CPVarImpl> impl;
  CPVar() = default;
  explicit CPVar(llvm::StringRef name, VarType type,
                 std::optional<double> lowerBound = /* -inf */ std::nullopt,
                 std::optional<double> upperBound = /* +inf */ std::nullopt)
      : impl(std::make_shared<detail::CPVarImpl>(name, type, lowerBound,
                                                 upperBound)) {};
  bool operator<(const CPVar &other) const noexcept;
  std::string getName();
};

inline std::pair<CPVar, CPVar> makeSortedPair(const CPVar &a, const CPVar &b) {
  return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

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

LinExpr operator+(const CPVar &left, double right);
LinExpr operator+(double left, const CPVar &right);
LinExpr operator-(const CPVar &left, double right);
LinExpr operator-(double left, const CPVar &right);
LinExpr operator+(const CPVar &left, const CPVar &right);
LinExpr operator-(const CPVar &left, const CPVar &right);
LinExpr operator+(const LinExpr &left, double right);
LinExpr operator+(double left, const LinExpr &right);
LinExpr operator+(const LinExpr &left, const LinExpr &right);
void operator+=(LinExpr &left, const LinExpr &right);
LinExpr operator-(const LinExpr &left, const LinExpr &right);
void operator-=(LinExpr &left, const LinExpr &right);

LinExpr operator*(double c, const CPVar &v);
LinExpr operator*(const CPVar &v, double c);
LinExpr operator*(double c, const LinExpr &expr);
LinExpr operator*(const LinExpr &expr, double c);

struct QuadExpr {
  LinExpr linexpr;
  std::map<std::pair<CPVar, CPVar>, double> quadTerms;
  QuadExpr() = default;
  QuadExpr(double value) { linexpr = LinExpr(value); }
  QuadExpr(const CPVar &var) { linexpr = LinExpr(var); }
  QuadExpr(const LinExpr &expr) { linexpr = expr; }
};

QuadExpr operator*(const LinExpr &lhs, const LinExpr &rhs);
QuadExpr operator*(double lhs, const QuadExpr &rhs);
QuadExpr operator+(const QuadExpr &lhs, const QuadExpr &rhs);
QuadExpr operator-(const QuadExpr &lhs, const QuadExpr &rhs);
void operator+=(QuadExpr &lhs, const QuadExpr &rhs);
void operator-=(QuadExpr &lhs, const QuadExpr &rhs);

enum Predicate {
  /* <= */ LE,
  /* == */ EQ
};

/// Class for constraints
/// It has the form:
/// [expr] [pred] 0
/// For example:
/// - x + 2 * y - z - 1 <= 0
/// - x + 2 * y - z + 1 <= 0
/// - x + 2 * y - z + 2 == 0
/// The rhs is always 0

// NOTE: this name is borrowed from gurobi (GRBTempConstr)
struct TempConstr {
  // The LHS expression. RHS is omitted because it is always set to zero.
  QuadExpr expr;
  Predicate pred;
};

TempConstr operator<=(const QuadExpr &lhs, const QuadExpr &rhs);
TempConstr operator>=(const QuadExpr &lhs, const QuadExpr &rhs);
TempConstr operator==(const QuadExpr &lhs, const QuadExpr &rhs);

/// Abstract base class for different solvers (e.g., Gurobi, Google's solvers,
/// or Cbc).
///
/// This is overloaded for the Gurobi solver and the Google's OR Tools API.
///
/// The user can use exactly the same API to define variables, add
/// constraints/objectives.
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
#ifdef DYNAMATIC_ENABLE_CBC
    CBC,
#endif      // DYNAMATIC_ENABLE_CBC
    DEFAULT // Dummy option
  };
  SolverKind solverKind;
  SolverKind getKind() const { return solverKind; }
  static inline bool classof(CPSolver const *) { return true; }
  // [END LLVM RTTI prerequisites]

  // Solver timeout in second.
  // If timeout <= 0, then this option is ignored.
  int timeout;

  // Maximum number of threads used in MILP solving
  int maxThreads;
  CPSolver(int timeout, SolverKind k, int maxThreads)
      : solverKind(k), timeout(timeout), maxThreads(maxThreads) {}

  virtual ~CPSolver() = default;
  // Virtual class methods: they provide a unified interface for all available
  // solvers.
  virtual CPVar addVar(const CPVar &var) = 0;
  // Create var, add gurobi var, and then return the created variable
  virtual CPVar addVar(const std::string &name, VarType type,
                       std::optional<double> lb, std::optional<double> ub) = 0;
  virtual void addConstr(const TempConstr &constraint,
                         llvm::StringRef constrName) = 0;
  void addConstr(const TempConstr &constraint) { addConstr(constraint, ""); }
  virtual void addQConstr(const TempConstr &constraint,
                          llvm::StringRef constrName) = 0;
  virtual void setMaximizeObjective(const LinExpr &expr) = 0;
  virtual void optimize() = 0;
  virtual double getValue(const CPVar &var) const = 0;
  virtual double getObjective() const = 0;

  virtual void write(llvm::StringRef filePath) const = 0;

  virtual void writeSol(llvm::StringRef filePath) const = 0;

  std::string symbolizeStatus() {
    if (status == OPTIMAL)
      return "OPTIMAL";
    if (status == NONOPTIMAL)
      return "NONOPTIMAL";
    if (status == INFEASIBLE)
      return "INFEASIBLE";
    if (status == UNBOUNDED)
      return "UNBOUNDED";
    if (status == UNKNOWN)
      return "UNKNOWN";
    return "ERROR";
  }
};

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
class GurobiSolver : public CPSolver {

  std::unique_ptr<GRBEnv> env;
  std::map<CPVar, GRBVar> variables;
  std::unique_ptr<GRBModel> model;

  // Track the added names: prevent adding variables with duplicated names
  std::set<std::string> names;

public:
  GurobiSolver(int timeout = -1 /* default = no timeout*/,
               int maxThreads = 4 /* default = maximum 4 threads */);

  CPVar addVar(const CPVar &var) override;

  // Create var, add gurobi var, and then return
  CPVar addVar(const std::string &name, VarType type, std::optional<double> lb,
               std::optional<double> ub) override;
  void addConstr(const TempConstr &constraint,
                 llvm::StringRef constrName) override;
  void addQConstr(const TempConstr &constraint,
                  llvm::StringRef constrName) override;
  void optimize() override;

  void setMaximizeObjective(const LinExpr &expr) override;

  void write(llvm::StringRef filePath) const override {
    model->write(filePath.str());
  }
  void writeSol(llvm::StringRef filePath) const override;

  /// Retrieve the value from the solved MILP
  ///
  /// Example:
  /// auto resultMyVar = solver.getValue(myVar);
  double getValue(const CPVar &var) const override;

  double getObjective() const override;

  // [START LLVM RTTI prerequisites]
  static bool classof(const CPSolver *b) { return b->getKind() == GUROBI; }
  static bool classof(const GurobiSolver *b) { return true; }
  // [END LLVM RTTI prerequisites]
};
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#ifdef DYNAMATIC_ENABLE_CBC

class CbcSolver : public CPSolver {

  OsiClpSolverInterface solver;
  std::map<CPVar, int> variables; // map Var -> column index
  std::set<std::string> names;

public:
  CbcSolver(int timeout = -1 /* default = no timeout */,
            int maxThreads = -1 /* note: currently this option has no effect */)
      : CPSolver(timeout, CBC, maxThreads) {
    // Suppress the solver's output
    solver.messageHandler()->setLogLevel(-1);
    solver.getModelPtr()->messageHandler()->setLogLevel(-1);
  }

  CPVar addVar(const CPVar &var) override;
  CPVar addVar(const std::string &name, VarType type, std::optional<double> lb,
               std::optional<double> ub) override;
  void addConstr(const TempConstr &constraint,
                 llvm::StringRef constrName) override;

  void addQConstr(const TempConstr &constraint, llvm::StringRef name) override {
    llvm::report_fatal_error(
        "Quadratic constraints is currently unavailable for CBC!");
  }
  void setMaximizeObjective(const LinExpr &expr) override;
  void optimize() override;
  void write(llvm::StringRef filePath) const override {
    // HACK: This implementation bypasses the log level and prints some warning
    // messages to stdout; this pollutes the final mlir output.
    //
    // Therefore, this write command currently doesn't do anything.
    //
    // solver.writeLp(filePath.str().c_str());
  }

  void writeSol(llvm::StringRef filePath) const override;

  double getValue(const CPVar &var) const override;

  double getObjective() const override;

  // [START LLVM RTTI prerequisites]
  static bool classof(const CbcSolver *b) { return true; }
  static bool classof(const CPSolver *b) { return b->getKind() == CBC; }
  // [END LLVM RTTI prerequisites]
};

#endif // DYNAMATIC_ENABLE_CBC

} // namespace dynamatic
