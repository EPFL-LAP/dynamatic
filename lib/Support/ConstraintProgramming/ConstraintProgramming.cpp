#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>

using namespace dynamatic;

// -------------------------------------------------------------
// Utility functions
// -------------------------------------------------------------

namespace dynamatic {
namespace detail {

// Avoid cyclic dependency in the overloading.
static LinExpr addVarConstImpl(const CPVar &left, double right) {
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

// Avoid cyclic dependency in the overloading.
static LinExpr subVarConstImpl(const CPVar &left, double right) {

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

static LinExpr mulVarConstImpl(const CPVar &left, double right) {
  LinExpr newExpr(left);
  newExpr.terms[left] *= right;
  newExpr.constant *= right;
  return newExpr;
}
} // namespace detail

// -------------------------------------------------------------
// Operator overloading implementation for the LP DSL
// ------------------------------------------------------------

bool CPVar::operator<(const CPVar &other) const noexcept {
  return this->impl->name < other.impl->name;
}

std::string CPVar::getName() { return this->impl->name; }

LinExpr operator+(const CPVar &left, double right) {
  return detail::addVarConstImpl(left, right);
}

LinExpr operator+(double left, const CPVar &right) {
  return detail::addVarConstImpl(right, left);
}

LinExpr operator-(const CPVar &left, double right) {
  return detail::subVarConstImpl(left, right);
}

LinExpr operator-(double left, const CPVar &right) {
  return -detail::subVarConstImpl(right, left);
}

LinExpr operator+(const CPVar &left, const CPVar &right) {
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

LinExpr operator-(const CPVar &left, const CPVar &right) {
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

LinExpr operator+(const LinExpr &left, double right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  newExpr.constant += right;
  return newExpr;
}

LinExpr operator+(double left, const LinExpr &right) {
  // REMARK:
  // this is a deep copy of "left"
  return (right + left);
}

LinExpr operator+(const LinExpr &left, const LinExpr &right) {
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

void operator+=(LinExpr &left, const LinExpr &right) {
  for (auto &[var, coeff] : right.terms) {
    left.terms[var] += coeff;
  }
  left.constant += right.constant;
}

LinExpr operator-(const LinExpr &left, const LinExpr &right) {
  LinExpr newExpr = left;
  for (auto &[var, coeff] : right.terms) {
    newExpr.terms[var] -= coeff;
  }
  newExpr.constant -= right.constant;
  return newExpr;
}

void operator-=(LinExpr &left, const LinExpr &right) {
  for (auto &[var, coeff] : right.terms) {
    left.terms[var] -= coeff;
  }
  left.constant -= right.constant;
}

LinExpr operator*(double c, const CPVar &v) {
  return detail::mulVarConstImpl(v, c);
}

LinExpr operator*(const CPVar &v, double c) {
  return detail::mulVarConstImpl(v, c);
}

LinExpr operator*(double c, const LinExpr &expr) {
  LinExpr newExpr(expr);
  for (auto &[var, coeff] : newExpr.terms) {
    newExpr.terms[var] = c * coeff;
  }
  newExpr.constant *= c;
  return newExpr;
}
LinExpr operator*(const LinExpr &expr, double c) { return c * expr; }

QuadExpr operator*(const LinExpr &lhs, const LinExpr &rhs) {
  QuadExpr e;
  // Quadratic terms:
  for (auto &[lhsTerm, lhsCoeff] : lhs.terms) {
    for (auto &[rhsTerm, rhsCoeff] : rhs.terms) {
      e.quadTerms[makeSortedPair(lhsTerm, rhsTerm)] += lhsCoeff * rhsCoeff;
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

QuadExpr operator*(double lhs, const QuadExpr &rhs) {
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

QuadExpr operator+(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = lhs;
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    e.quadTerms[quadTerm] += coeff;
  }
  // Linear and constant terms:
  e.linexpr = e.linexpr + rhs.linexpr;
  return e;
}

QuadExpr operator-(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = lhs;
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    e.quadTerms[quadTerm] -= coeff;
  }
  // Linear and constant terms:
  e.linexpr = e.linexpr - rhs.linexpr;
  return e;
}

void operator+=(QuadExpr &lhs, const QuadExpr &rhs) {
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    lhs.quadTerms[quadTerm] += coeff;
  }
  // Linear and constant terms:
  lhs.linexpr = lhs.linexpr + rhs.linexpr;
}

void operator-=(QuadExpr &lhs, const QuadExpr &rhs) {
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    lhs.quadTerms[quadTerm] -= coeff;
  }
  // Linear and constant terms:
  lhs.linexpr = lhs.linexpr - rhs.linexpr;
}

// -------------------------------------------------------------
// TempConstr method implementations
// ------------------------------------------------------------

TempConstr operator<=(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = (lhs - rhs);
  TempConstr c;
  c.expr = e;
  c.pred = LE;
  return c;
}

TempConstr operator>=(const QuadExpr &lhs, const QuadExpr &rhs) {
  return (rhs <= lhs);
}

TempConstr operator==(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = (lhs - rhs);
  TempConstr c;
  c.expr = e;
  c.pred = EQ;
  return c;
}

// -------------------------------------------------------------
// GurobiSolver method implementations
// ------------------------------------------------------------

GurobiSolver::GurobiSolver(int timeout, int maxThreads)
    : CPSolver(timeout, GUROBI, maxThreads) {
  env = std::make_unique<GRBEnv>(true);

  // Suppress outputs to stdout (clashes with the MLIR output file).
  env->set(GRB_IntParam_OutputFlag, 0);

  // Always use the same random seed to make the solution deterministic.
  env->set(GRB_IntParam::GRB_IntParam_Seed, 0);

  if (timeout > 0) {
    env->set(GRB_DoubleParam_TimeLimit, timeout);
  }

  if (maxThreads > 0) {
    env->set(GRB_IntParam_Threads, maxThreads);
  }

  env->start();

  model = std::make_unique<GRBModel>(*env);
}

void GurobiSolver::addConstr(const TempConstr &constraint,
                             llvm::StringRef constrName) {
  if (!constraint.expr.quadTerms.empty())
    llvm::report_fatal_error(
        "Adding a linear constraint with quadratic terms!");

  GRBLinExpr expr = 0;
  // Linear terms
  for (auto &[name, coeff] : constraint.expr.linexpr.terms) {
    expr += coeff * variables[name];
  }

  // Constant terms
  expr += constraint.expr.linexpr.constant;
  if (constraint.pred == LE)
    model->addConstr(expr <= 0, constrName.str());
  else if (constraint.pred == EQ)
    model->addConstr(expr == 0, constrName.str());
  else
    llvm_unreachable("Unknown predicate!");
}

void GurobiSolver::addQConstr(const TempConstr &constraint,
                              llvm::StringRef constrName) {
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

CPVar GurobiSolver::addVar(const std::string &name, VarType type,
                           std::optional<double> lb, std::optional<double> ub) {
  auto var = CPVar(name, type, lb, ub);
  return addVar(var);
}

CPVar GurobiSolver::addVar(const CPVar &var) {
  if (names.count(var.impl->name)) {
    llvm::report_fatal_error("Adding variable with duplicated names is not "
                             "permitted! Aborting...");
  }
  double lb = var.impl->lowerBound.value_or(-GRB_INFINITY);
  double ub = var.impl->upperBound.value_or(GRB_INFINITY);
  char type;
  switch (var.impl->type) {
  case REAL:
    type = GRB_CONTINUOUS;
    break;
  case INTEGER:
    type = GRB_INTEGER;
    break;
  case BOOLEAN:
    type = GRB_BINARY;
  }
  variables[var] = model->addVar(lb, ub, 0.0, type, var.impl->name);
  names.insert(var.impl->name);
  return var;
}

void GurobiSolver::setMaximizeObjective(const LinExpr &expr) {
  GRBLinExpr obj = 0;
  for (auto &[name, coeff] : expr.terms) {
    obj += coeff * variables[name];
  }
  obj += expr.constant;
  // NOTE: the constant term can be ignored in the objective
  model->setObjective(obj, GRB_MAXIMIZE);
}

void GurobiSolver::optimize() {
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
  case GRB_INF_OR_UNBD:
    status = UNBOUNDED;
    break;
  case GRB_INFEASIBLE:
    status = INFEASIBLE;
    break;
  default:
    status = ERROR;
  }
}

void GurobiSolver::writeSol(llvm::StringRef filePath) const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Calling writeSol before the model was solved!");
  }

  std::ofstream myfile(filePath.str());
  if (myfile.is_open()) {
    for (auto &[var, _] : variables) {
      myfile << var.impl->name << " = " << getValue(var) << "\n";
    }
  } else {
    llvm::errs() << "Unable to open file: " << filePath << "!\n";
    llvm::report_fatal_error("Unable to open file!");
  }
}

double GurobiSolver::getValue(const CPVar &var) const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::errs() << "Solution is not available while retrieving "
                 << var.impl->name << "!\n";
    llvm::report_fatal_error("Cannot retrieve the value of variable!");
  }
  return variables.at(var).get(GRB_DoubleAttr_X);
}

double GurobiSolver::getObjective() const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Cannot retrieve the objective because the "
                             "solution is not available!");
  }
  return model->get(GRB_DoubleAttr_ObjVal);
}

// -------------------------------------------------------------
// CbcSolver method implementations
// ------------------------------------------------------------

CPVar CbcSolver::addVar(const CPVar &var) {
  if (names.count(var.impl->name)) {
    llvm::report_fatal_error("Adding variable with duplicated names is not "
                             "permitted! Aborting...");
  }

  double lb = var.impl->lowerBound.value_or(-1e20);
  double ub = var.impl->upperBound.value_or(1e20);

  int colIndex = solver.getNumCols();

  // Add an empty column for this variable
  solver.addCol(0, nullptr, nullptr, lb, ub, 0.0);
  variables[var] = colIndex;

  // Set variable type
  if (var.impl->type == INTEGER)
    solver.setInteger(colIndex);
  else if (var.impl->type == BOOLEAN) {
    solver.setInteger(colIndex);
    solver.setColUpper(colIndex, 1.0);
    solver.setColLower(colIndex, 0.0);
  } // REAL is default

  names.insert(var.impl->name);
  return var;
}

CPVar CbcSolver::addVar(const std::string &name, VarType type,
                        std::optional<double> lb, std::optional<double> ub) {
  auto var = CPVar(name, type, lb, ub);
  return addVar(var);
}

double CbcSolver::getValue(const CPVar &var) const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::errs() << "Solution is not available while retrieving "
                 << var.impl->name << "!\n";
    llvm::report_fatal_error("Cannot retrieve the value of variable!");
  }
  return solver.getColSolution()[variables.at(var)];
}

void CbcSolver::writeSol(llvm::StringRef filePath) const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Calling writeSol before the model was solved!");
  }

  std::ofstream solLogFile(filePath.str());
  if (solLogFile.is_open()) {
    for (auto &[var, _] : variables) {
      solLogFile << var.impl->name << " = " << getValue(var) << "\n";
    }
  } else {
    llvm::errs() << "Unable to open file: " << filePath << "!\n";
    llvm::report_fatal_error("Unable to open file!");
  }
}

void CbcSolver::optimize() {
  CbcModel model(solver);
  // Disable all output
  // 0 = no output, 1 = minimal, higher = more verbose
  model.setLogLevel(-1);

  if (this->timeout > 0) {
    model.setMaximumSeconds(timeout);
  }

  model.setRandomSeed(0);

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

double CbcSolver::getObjective() const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Cannot retrieve the objective because the "
                             "solution is not available!");
  }
  return solver.getObjValue();
}

void CbcSolver::setMaximizeObjective(const LinExpr &expr) {
  // Create objective array
  std::vector<double> obj(solver.getNumCols(), 0.0);
  for (auto &[var, coeff] : expr.terms) {
    // Set objective in solver
    solver.setObjCoeff(variables.at(var), coeff);
  }

  // CBC minimizes by default; for maximize, multiply by -1
  solver.setObjSense(-1.0);
}

void CbcSolver::addConstr(const TempConstr &constraint,
                          llvm::StringRef constrName) {
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

} // namespace dynamatic
