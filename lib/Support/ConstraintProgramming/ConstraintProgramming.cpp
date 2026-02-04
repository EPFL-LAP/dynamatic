#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "dynamatic/Support/System.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace dynamatic;

// -------------------------------------------------------------
// Utility functions
// -------------------------------------------------------------

#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

namespace {
bool containsInvalid(StringRef filePath) {
  // 1. Open the file
  auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (std::error_code ec = bufferOrErr.getError())
    return false;

  // 2. Access the buffer as a StringRef
  llvm::StringRef content = bufferOrErr.get()->getBuffer();

  // 3. Search for both variations
  // .contains() is available in newer LLVM; otherwise use .find() != npos
  return content.contains("invalid") || content.contains("Invalid");
}
} // namespace

static unsigned int modelCount = 0;

namespace dynamatic {
namespace detail {

LogicalResult CbcSoluParser::parseSolverOutput(StringRef soluFileName) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(soluFileName);
  if (!bufferOrErr)
    return failure();

  llvm::line_iterator it(*bufferOrErr->get(), /*SkipBlanks=*/true);

  // NOTE: The default constructor of llvm::line_iterator is the "end" iterator
  llvm::line_iterator end;

  // ---- First line: status + objective value ----
  if (it == end)
    return failure();

  {
    // Example:
    // "Optimal - objective value 999.00000000"
    StringRef line = *it;

    if (line.startswith("Optimal")) {
      status = CPSolver::Status::OPTIMAL;
    } else if (line.startswith("Infeasible")) {
      status = CPSolver::Status::INFEASIBLE;
    } else if (line.startswith("Unbounded")) {
      status = CPSolver::Status::UNBOUNDED;
    } else {
      status = CPSolver::Status::UNKNOWN;
    }

    // Extract objective value
    auto pos = line.find("objective value");
    if (pos != StringRef::npos) {
      StringRef valueStr =
          line.drop_front(pos + strlen("objective value")).trim();
      valueStr.getAsDouble(objectiveValue);
    }
  }

  ++it;

  // ---- Remaining lines: variable assignments ----
  for (; it != end; ++it) {
    // Example:
    // "1 numExec_times_sArc_1_1             999                       1"

    std::stringstream ss(it->str());
    int index;
    std::string varName;
    double value;

    ss >> index >> varName >> value;
    if (ss.fail())
      continue;

    results[varName] = value;
  }

  return success();
}

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

std::string CPVar::getName() const { return this->impl->name; }

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

static std::string formatCoeffAndName(double coeff, const CPVar &v) {
  std::stringstream ss;
  if (std::abs(coeff) == 1.0) {
    return v.impl->name;
  }
  ss << std::fixed << std::abs(coeff) << " " + v.impl->name;
  return ss.str();
}

std::string LinExpr::writeLp() const {
  std::stringstream ss;
  ss << std::fixed;
  unsigned count = 0;
  for (auto &[term, coeff] : this->terms) {
    llvm::errs() << "[DEBUG] name " << term.impl->name << "\n";
    if (coeff == 0.0)
      continue;
    if (count == 0) {
      ss << (coeff > 0 ? "" : "- ") << formatCoeffAndName(coeff, term);
    } else {
      ss << (coeff > 0 ? " + " : " - ") << formatCoeffAndName(coeff, term);
    }
    count += 1;
  }
  if (0.0 != std::abs(constant))
    ss << (this->constant > 0 ? " + " : " - ") << std::abs(this->constant);
  return ss.str();
}

std::string TempConstr::writeLp() const {
  if (!this->expr.quadTerms.empty())
    llvm::report_fatal_error("Quandratic formula is unsupported!");

  LinExpr linexpr = this->expr.linexpr;

  std::stringstream ss;
  ss << std::fixed;
  unsigned count = 0;
  for (auto &[term, coeff] : linexpr.terms) {
    if (coeff == 0.0)
      continue;
    if (count == 0) {
      ss << (coeff > 0 ? "" : "- ") << formatCoeffAndName(coeff, term);
    } else {
      ss << (coeff > 0 ? " + " : " - ") << formatCoeffAndName(coeff, term);
    }
    count += 1;
  }

  if (this->pred == EQ)
    ss << " = ";
  else if (this->pred == LE)
    ss << " <= ";
  else
    llvm::report_fatal_error("unknown predicate");

  // TODO: not sure if we can put the constant on the LHS.
  // Put the constant in the RHS and flip the sign
  ss << (linexpr.constant > 0 ? " -" : "") << std::abs(linexpr.constant);
  return ss.str();
}

// -------------------------------------------------------------
// GurobiSolver method implementations
// ------------------------------------------------------------

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED

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

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

// -------------------------------------------------------------
// CbcSolver method implementations
// ------------------------------------------------------------

#ifdef DYNAMATIC_ENABLE_CBC

CPVar CbcSolver::addVar(const CPVar &var) {
  if (names.count(var.impl->name)) {
    llvm::report_fatal_error("Adding variable with duplicated names is not "
                             "permitted! Aborting...");
  }
  names.insert(var.impl->name);
  variables.insert(var);
  return var;
}

CPVar CbcSolver::addVar(const std::string &name, VarType type,
                        std::optional<double> lb, std::optional<double> ub) {
  auto var = CPVar(name, type, lb, ub);
  return addVar(var);
}

double CbcSolver::getValue(const CPVar &var) const {
  llvm::errs() << "[DEBUG]!! " << var.impl->name << "\n";
  // This means some internal naming is malformed. Should not be user's fault.
  if (!this->solution.results.count(var.impl->name)) {
    return 0.0;
  }
  return this->solution.results.at(var.impl->name);
}

void CbcSolver::writeSol(llvm::StringRef filePath) const {
  // Update solver with solution for getValue
  std::ofstream myfile(filePath.str());
  if (myfile.is_open()) {
    for (const auto &var : variables) {
      myfile << var.impl->name << " = " << getValue(var) << "\n";
    }

    myfile << this->getObjective() << "\n";
  } else {
    llvm::errs() << "Unable to open file: " << filePath << "!\n";
    llvm::report_fatal_error("Unable to open file!");
  }
}

void CbcSolver::writeLp(llvm::StringRef filepath) const {

  std::error_code ec;
  llvm::raw_fd_ostream os(filepath, ec);

  if (ec) {
    llvm::errs() << "Error opening file: " << ec.message() << "\n";
    return;
  }
  os << "Maximize\n";
  os << this->maxObjective.writeLp() << "\n";

  os << "Subject to\n";
  for (const auto &[name, constr] : constraints) {
    os << constr.writeLp() << "\n";
  }
  os << "\n\n";

  unsigned numBool = 0, numInt = 0, numReal = 0;

  for (const auto &v : variables) {
    if (v.impl->type == BOOLEAN)
      ++numBool;
    else if (v.impl->type == INTEGER)
      ++numInt;
    else if (v.impl->type == REAL)
      ++numReal;
    else
      // Shouldn't be caused by invalid user input
      llvm_unreachable("Unknown type");
  }

  // Print the bounds:
  for (const auto &v : variables) {
    if (v.impl->type == BOOLEAN)
      continue;
    if (v.impl->upperBound) {
      os << v.impl->name << " <= " << llvm::format("%.2f", *v.impl->upperBound)
         << "\n";
    }
    if (v.impl->lowerBound) {
      os << v.impl->name << " >= " << llvm::format("%.2f", *v.impl->lowerBound)
         << "\n";
    }
  }

  // Declare the data types
  if (numBool) {
    os << "Binary\n";
    for (const auto &v : variables) {
      if (v.impl->type == BOOLEAN)
        os << v.impl->name << " ";
    }
    os << "\n\n";
  }

  if (numInt) {
    os << "General\n";
    for (const auto &v : variables) {
      if (v.impl->type == INTEGER)
        os << v.impl->name << " ";
    }
    os << "\n\n";
  }

  // if (numReal) {
  //   os << "Real\n";
  //   for (const auto &v : variables) {
  //     if (v.impl->type == REAL)
  //       os << v.impl->name << " ";
  //   }
  //   os << "\n\n";
  // }

  os << "End\n";
}

void CbcSolver::optimize() {
  std::string lpFile = llvm::formatv("cbc_model_{0}.lp", modelCount);
  std::string solFile = llvm::formatv("cbc_solution_{0}.sol", modelCount);
  std::string redirectFile = llvm::formatv("cbc_output_{0}.log", modelCount);
  modelCount += 1;
  this->writeLp(lpFile);

  std::string errMsg;
  bool executionFailed;
  // Find the program in the PATH
  auto programName = llvm::sys::findProgramByName("cbc");
  std::error_code ec = programName.getError();
  if (ec) {
    llvm::report_fatal_error("Could not find cbc in the path!\n");
  }

  std::vector<std::optional<StringRef>> redirects = {std::nullopt, redirectFile,
                                                     std::nullopt};

  int exitCode;

  if (this->timeout > 0)
    exitCode = llvm::sys::ExecuteAndWait(
        *programName,
        {"cbc", lpFile, "sec", std::to_string(timeout), "solve", "solu",
         solFile},
        std::nullopt, redirects, 0, 0, &errMsg, &executionFailed);
  else
    exitCode = llvm::sys::ExecuteAndWait(
        *programName, {"cbc", lpFile, "solve", "solu", solFile}, std::nullopt,
        redirects, 0, 0, &errMsg, &executionFailed);

  if (exitCode != 0) {
    llvm::errs() << "Cbc failed with exit code " << exitCode << "!\n";
    llvm::report_fatal_error("Cbc failed!");
  }

  if (containsInvalid(redirectFile)) {
    std::string errMsg = "The file " + lpFile +
                         " is invalid for Cbc! See the logfile" + redirectFile;
    llvm::report_fatal_error(StringRef(errMsg));
  }

  if (auto res = this->solution.parseSolverOutput(solFile); failed(res))
    llvm::report_fatal_error("The solution is malformed!\n");

  this->status = this->solution.status;
}

double CbcSolver::getObjective() const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Cannot retrieve the objective because the "
                             "solution is not available!");
  }

  return this->solution.objectiveValue;
}

void CbcSolver::setMaximizeObjective(const LinExpr &expr) {
  this->maxObjective = expr;
}

void CbcSolver::addConstr(const TempConstr &constraint,
                          llvm::StringRef constrName) {

  this->constraints.emplace_back(constrName, constraint);
}
#endif // DYNAMATIC_ENABLE_CBC

} // namespace dynamatic
