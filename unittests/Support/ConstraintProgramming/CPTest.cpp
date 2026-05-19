#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace dynamatic;

namespace {

// Type-erased solver factory
using SolverFactory = std::function<std::unique_ptr<CPSolver>()>;

// Parameterized test fixture
class ParamSolverTest : public ::testing::TestWithParam<SolverFactory> {};

TEST_P(ParamSolverTest, basicMILPTest) {

  // [Using our API to solve the result]
  auto solver = GetParam()();
  auto x = solver->addVar("x", REAL, /* lb */ 0, std::nullopt);
  auto y = solver->addVar("y", REAL, /* lb */ 0, std::nullopt);
  solver->addConstr(x + 2 * y <= 14);
  solver->addConstr(3 * x - y >= 0);
  solver->addConstr(x - y <= 2);
  solver->setMaximizeObjective(3 * x + 4 * y);
  solver->optimize();

  // Check: The result from our API and from Gurobi's API must be exactly the
  // same.
  EXPECT_NEAR(34, solver->getObjective(), 1e-6);
  EXPECT_NEAR(6, solver->getValue(x), 1e-6);
  EXPECT_NEAR(4, solver->getValue(y), 1e-6);
}

TEST_P(ParamSolverTest, basicIntegerProgramTest) {
  // [Using our API to solve the result]
  auto solver = GetParam()();
  auto x = solver->addVar("x", INTEGER, /* lb */ 0,
                          /* infinity */ std::nullopt);
  auto y = solver->addVar("y", INTEGER, /* lb */ 0,
                          /* infinity */ std::nullopt);
  auto z = solver->addVar("z", INTEGER, /* lb */ 0,
                          /* infinity */ std::nullopt);
  solver->addConstr(2 * x + 7 * y + 3 * z <= 50);
  solver->addConstr(3 * x + 5 * y + 7 * z <= 45);
  solver->addConstr(5 * x + 2 * y - 6 * z <= 37);
  solver->setMaximizeObjective(2 * x + 2 * y + 3 * z);
  solver->optimize();

  // [Using Gurobi's API to solve the result]
  GRBEnv env = GRBEnv(true);
  env.start();
  GRBModel model = GRBModel(env);
  //
  // Create variables x and y (continuous by default)
  GRBVar a = model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER, "a");
  GRBVar b = model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER, "b");
  GRBVar c = model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER, "c");

  model.addConstr(2 * a + 7 * b + 3 * c <= 50);
  model.addConstr(3 * a + 5 * b + 7 * c <= 45);
  model.addConstr(5 * a + 2 * b - 6 * c <= 37);

  model.setObjective(2 * a + 2 * b + 3 * c, GRB_MAXIMIZE);
  model.optimize();

  // Check: The result from our API and from Gurobi's API must be exactly the
  // same.
  EXPECT_EQ(25, solver->getObjective());
  EXPECT_EQ(8, solver->getValue(x));
  EXPECT_EQ(0, solver->getValue(y));
  EXPECT_EQ(3, solver->getValue(z));
}

// [START AI-generated test cases]

TEST_P(ParamSolverTest, SimpleMaxLP) {
  auto solver = GetParam()();

  auto x = solver->addVar("x", REAL, 0, std::nullopt);
  auto y = solver->addVar("y", REAL, 0, std::nullopt);

  solver->addConstr(x + y <= 10);
  solver->setMaximizeObjective(x + 2 * y);
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);

  EXPECT_LE(xVal + yVal, 10 + 1e-6); // Constraint check
}

TEST_P(ParamSolverTest, EqualityConstraintLP) {
  auto solver = GetParam()();

  auto x = solver->addVar("x", REAL, 0, std::nullopt);
  auto y = solver->addVar("y", REAL, 0, std::nullopt);

  solver->addConstr(x + y == 10);
  solver->setMaximizeObjective(2 * x + y);
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);

  EXPECT_NEAR(xVal + yVal, 10, 1e-6);
}

TEST_P(ParamSolverTest, BoundedVariablesLP) {
  auto solver = GetParam()();

  auto x = solver->addVar("x", REAL, 0, 5);
  auto y = solver->addVar("y", REAL, 0, 5);

  solver->addConstr(x + y >= 6);
  solver->setMaximizeObjective(-(x + y)); // Minimize
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);

  EXPECT_TRUE(xVal >= -1e-6 && xVal <= 5 + 1e-6);
  EXPECT_TRUE(yVal >= -1e-6 && yVal <= 5 + 1e-6);
  EXPECT_TRUE(xVal + yVal >= 6 - 1e-6);
}

TEST_P(ParamSolverTest, RedundantConstraintLP) {
  auto solver = GetParam()();

  auto x = solver->addVar("x", REAL, 0, std::nullopt);

  solver->addConstr(x >= 2);
  solver->addConstr(x >= 1); // Redundant

  solver->setMaximizeObjective(-LinExpr(x)); // Minimize x
  solver->optimize();

  auto xVal = solver->getValue(x);

  EXPECT_TRUE(xVal >= 2 - 1e-6);
}

TEST_P(ParamSolverTest, MixedIntegerLP) {
  auto solver = GetParam()();

  auto x = solver->addVar("x", INTEGER, 0, std::nullopt);
  auto y = solver->addVar("y", REAL, 0, std::nullopt);

  solver->addConstr(x + y >= 4);
  solver->addConstr(2 * x + y <= 10);

  solver->setMaximizeObjective(x + 2 * y);
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);

  EXPECT_NEAR(xVal, std::round(xVal), 1e-6); // integer check
  EXPECT_TRUE(xVal + yVal >= 4 - 1e-6);
  EXPECT_TRUE(2 * xVal + yVal <= 10 + 1e-6);
}

TEST_P(ParamSolverTest, SimpleMinLP) {
  auto solver = GetParam()();

  auto x = solver->addVar("x", REAL, 1, std::nullopt);
  auto y = solver->addVar("y", REAL, 0, std::nullopt);

  solver->addConstr(2 * x + y >= 5);
  solver->setMaximizeObjective(-(x + y)); // Minimization via negation
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);

  EXPECT_TRUE(xVal * 2 + yVal >= 5 - 1e-6);
}

TEST_P(ParamSolverTest, SmallIntegerProgram) {
  auto solver = GetParam()();

  auto x = solver->addVar("x", INTEGER, 0, 5);
  auto y = solver->addVar("y", INTEGER, 0, 5);

  solver->addConstr(x + 2 * y <= 6);
  solver->setMaximizeObjective(x + y);
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);

  EXPECT_TRUE(xVal + 2 * yVal <= 6 + 1e-6);
}

TEST_P(ParamSolverTest, BigMConstraintCrossCheck) {
  auto solver = GetParam()();

  auto x = solver->addVar("x", REAL, 0, 10);
  auto y = solver->addVar("y", BOOLEAN, std::nullopt, std::nullopt);

  double bigConst = 1e6;
  // If y = 0 then x <= 3, else no restriction
  solver->addConstr(x - bigConst * y <= 3);

  solver->setMaximizeObjective(x);
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);
  auto objVal = solver->getObjective();

  // Solve with Gurobi for cross-check
  GRBEnv env(true);
  env.start();
  GRBModel model(env);

  GRBVar a = model.addVar(0, 10, 0.0, GRB_CONTINUOUS, "x");
  GRBVar b = model.addVar(0, 1, 0.0, GRB_BINARY, "y");

  model.addConstr(a - bigConst * b <= 3);

  model.setObjective(GRBLinExpr(a), GRB_MAXIMIZE);
  model.optimize();

  // Cross-check
  EXPECT_NEAR(10, xVal, 1e-6);
  EXPECT_NEAR(1, yVal, 1e-6);
  EXPECT_NEAR(10, objVal, 1e-6);
}

TEST(ExpressionOperators, LinExprPlusEquals) {
  CPVar x("x", REAL, 0, 10);
  CPVar y("y", REAL, 0, 10);

  LinExpr expr1 = x + 2 * y;
  LinExpr expr2 = y + 3;

  expr1 += expr2; // Should add expr2 to expr1

  EXPECT_DOUBLE_EQ(expr1.terms[x], 1.0); // x only in expr1
  EXPECT_DOUBLE_EQ(expr1.terms[y], 3.0); // 2 + 1
  EXPECT_DOUBLE_EQ(expr1.constant, 3.0); // 0 + 3
}

TEST(ExpressionOperators, LinExprMinusEquals) {
  CPVar x("x", REAL, 0, 10);
  CPVar y("y", REAL, 0, 10);

  LinExpr expr1 = 5 * x + 2 * y + 10;
  LinExpr expr2 = x + y + 3;

  expr1 -= expr2; // Should subtract expr2 from expr1

  EXPECT_DOUBLE_EQ(expr1.terms[x], 4.0); // 5 - 1
  EXPECT_DOUBLE_EQ(expr1.terms[y], 1.0); // 2 - 1
  EXPECT_DOUBLE_EQ(expr1.constant, 7.0); // 10 - 3
}

// Helper for comparing two LinExpr
static void expectExprEq(const LinExpr &lhs, const LinExpr &rhs) {
  EXPECT_EQ(lhs.terms.size(), rhs.terms.size());
  for (auto &[var, coeff] : lhs.terms) {
    auto it = rhs.terms.find(var);
    ASSERT_NE(it, rhs.terms.end());
    EXPECT_DOUBLE_EQ(it->second, coeff);
  }
  EXPECT_DOUBLE_EQ(lhs.constant, rhs.constant);
}

//----------------------------------------------------------------------------//
//  Tests for CPVar + double, double + CPVar, CPVar - double, double - CPVar
//----------------------------------------------------------------------------//

TEST(LinExprOpTest, VarPlusDouble) {
  CPVar x("x", REAL, std::nullopt, std::nullopt);
  LinExpr expr = x + 5.0;
  LinExpr expected;
  expected.terms[x] = 1.0;
  expected.constant = 5.0;
  expectExprEq(expr, expected);
}

TEST(LinExprOpTest, DoublePlusVar) {
  CPVar x("x", REAL, std::nullopt, std::nullopt);
  LinExpr expr = 5.0 + x;
  LinExpr expected;
  expected.terms[x] = 1.0;
  expected.constant = 5.0;
  expectExprEq(expr, expected);
}

TEST(LinExprOpTest, VarMinusDouble) {
  CPVar x("x", REAL, std::nullopt, std::nullopt);
  LinExpr expr = x - 3.0;
  LinExpr expected;
  expected.terms[x] = 1.0;
  expected.constant = -3.0;
  expectExprEq(expr, expected);
}

TEST(LinExprOpTest, DoubleMinusVar) {
  CPVar x("x", REAL, std::nullopt, std::nullopt);
  LinExpr expr = 7.0 - x;
  LinExpr expected;
  expected.terms[x] = -1.0;
  expected.constant = 7.0;
  expectExprEq(expr, expected);
}

//----------------------------------------------------------------------------//
//  Tests for combinations like double * CPVar + double, etc.
//----------------------------------------------------------------------------//

TEST(LinExprOpTest, DoubleTimesVarPlusDouble) {
  CPVar y("y", REAL, std::nullopt, std::nullopt);
  LinExpr expr = 2.5 * y + 1.5;
  LinExpr expected;
  expected.terms[y] = 2.5;
  expected.constant = 1.5;
  expectExprEq(expr, expected);
}

TEST(LinExprOpTest, VarTimesDoubleMinusDouble) {
  CPVar y("y", REAL, std::nullopt, std::nullopt);
  LinExpr expr = y * 2.0 - 4.0;
  LinExpr expected;
  expected.terms[y] = 2.0;
  expected.constant = -4.0;
  expectExprEq(expr, expected);
}

//----------------------------------------------------------------------------//
//  Tests for chained operations (associativity and correctness)
//----------------------------------------------------------------------------//

TEST(LinExprOpTest, ChainedAddSub) {
  CPVar x("x", REAL, std::nullopt, std::nullopt);
  CPVar y("y", REAL, std::nullopt, std::nullopt);
  LinExpr expr = 2.0 + x - 3.0 + y + 1.0;
  LinExpr expected;
  expected.terms[x] = 1.0;
  expected.terms[y] = 1.0;
  expected.constant = 0.0; // (2 - 3 + 1) = 0
  expectExprEq(expr, expected);
}

// [END AI-generated test cases]

// Factories for both solvers

#ifdef DYNAMATIC_ENABLE_CBC
std::unique_ptr<CPSolver> makeCbc() { return std::make_unique<CbcSolver>(); }
#endif

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
std::unique_ptr<CPSolver> makeGurobi() {
  return std::make_unique<GurobiSolver>();
}
#endif

#ifdef DYNAMATIC_ENABLE_CBC
// clang-format off
// Runs all MILP test with two different solvers
INSTANTIATE_TEST_SUITE_P(
  SolverImplementations,
  ParamSolverTest,
  ::testing::Values(
    makeCbc
  )
);
// clang-format on
#else
// Explicitly allow non-instantiated test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ParamSolverTest);
#endif

} // namespace
