#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include <gtest/gtest.h>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED

using namespace dynamatic::cp;

namespace {

// Type-erased solver factory
using SolverFactory = std::function<std::unique_ptr<CPSolver>()>;

// Parameterized test fixture
class ParamSolverTest : public ::testing::TestWithParam<SolverFactory> {};

TEST_P(ParamSolverTest, basicMILPTest) {

  // [Using our API to solve the result]
  // auto solver = GurobiSolver();
  auto solver = GetParam()();
  auto x = solver->addVariable("x", Var::REAL, /* lb */ 0, std::nullopt);
  auto y = solver->addVariable("y", Var::REAL, /* lb */ 0, std::nullopt);
  solver->addLinearConstraint(x + 2 * y <= 14);
  solver->addLinearConstraint(3 * x - y >= 0);
  solver->addLinearConstraint(x - y <= 2);
  solver->setMaximizeObjective(3 * x + 4 * y);
  solver->optimize();

  // [Using Gurobi's API to solve the result]
  GRBEnv env = GRBEnv(true);
  env.start();

  // Create an empty model
  GRBModel model = GRBModel(env);

  // Create variables x and y (continuous by default)
  GRBVar a = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x");
  GRBVar b = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "y");

  // Add constraints
  model.addConstr(a + 2 * b <= 14, "c1");
  model.addConstr(3 * a - b >= 0, "c2");
  model.addConstr(a - b <= 2, "c3");

  // Set objective: maximize 3*x + 4*y
  model.setObjective(3 * a + 4 * b, GRB_MAXIMIZE);

  // Optimize the model
  model.optimize();

  // Check: The result from our API and from Gurobi's API must be exactly the
  // same.
  EXPECT_EQ(model.get(GRB_DoubleAttr_ObjVal), solver->getObjective());
  EXPECT_EQ(a.get(GRB_DoubleAttr_X), solver->getValue(x));
  EXPECT_EQ(b.get(GRB_DoubleAttr_X), solver->getValue(y));
}

TEST_P(ParamSolverTest, basicIntegerProgramTest) {
  // [Using our API to solve the result]
  // auto solver = GurobiSolver();
  auto solver = GetParam()();
  auto x = solver->addVariable("x", Var::INTEGER, /* lb */ 0,
                               /* infinity */ std::nullopt);
  auto y = solver->addVariable("y", Var::INTEGER, /* lb */ 0,
                               /* infinity */ std::nullopt);
  auto z = solver->addVariable("z", Var::INTEGER, /* lb */ 0,
                               /* infinity */ std::nullopt);
  solver->addLinearConstraint(2 * x + 7 * y + 3 * z <= 50);
  solver->addLinearConstraint(3 * x + 5 * y + 7 * z <= 45);
  solver->addLinearConstraint(5 * x + 2 * y - 6 * z <= 37);
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
  EXPECT_EQ(model.get(GRB_DoubleAttr_ObjVal), solver->getObjective());
  EXPECT_EQ(a.get(GRB_DoubleAttr_X), solver->getValue(x));
  EXPECT_EQ(b.get(GRB_DoubleAttr_X), solver->getValue(y));
  EXPECT_EQ(c.get(GRB_DoubleAttr_X), solver->getValue(z));
}

// [START AI-generated test cases]

TEST_P(ParamSolverTest, SimpleMaxLP) {
  auto solver = GetParam()();

  auto x = solver->addVariable("x", Var::REAL, 0, std::nullopt);
  auto y = solver->addVariable("y", Var::REAL, 0, std::nullopt);

  solver->addLinearConstraint(x + y <= 10);
  solver->setMaximizeObjective(x + 2 * y);
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);
  auto obj = solver->getObjective();

  EXPECT_TRUE(xVal.has_value());
  EXPECT_TRUE(yVal.has_value());
  EXPECT_TRUE(obj.has_value());
  EXPECT_LE(xVal.value() + yVal.value(), 10 + 1e-6); // Constraint check
}

TEST_P(ParamSolverTest, SimpleMinLP) {
  auto solver = GetParam()();

  auto x = solver->addVariable("x", Var::REAL, 1, std::nullopt);
  auto y = solver->addVariable("y", Var::REAL, 0, std::nullopt);

  solver->addLinearConstraint(2 * x + y >= 5);
  solver->setMaximizeObjective(-(x + y)); // Minimization via negation
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);

  EXPECT_TRUE(xVal.value() * 2 + yVal.value() >= 5 - 1e-6);
}

TEST_P(ParamSolverTest, SmallIntegerProgram) {
  auto solver = GetParam()();

  auto x = solver->addVariable("x", Var::INTEGER, 0, 5);
  auto y = solver->addVariable("y", Var::INTEGER, 0, 5);

  solver->addLinearConstraint(x + 2 * y <= 6);
  solver->setMaximizeObjective(x + y);
  solver->optimize();

  auto xVal = solver->getValue(x);
  auto yVal = solver->getValue(y);

  EXPECT_TRUE(xVal.value() + 2 * yVal.value() <= 6 + 1e-6);
}

TEST_P(ParamSolverTest, BigMConstraintCrossCheck) {
  auto solver = GetParam()();

  auto x = solver->addVariable("x", Var::REAL, 0, 10);
  auto y = solver->addVariable("y", Var::BOOLEAN, std::nullopt, std::nullopt);

  double bigConst = 1e6;
  // If y = 0 then x <= 3, else no restriction
  solver->addLinearConstraint(x - bigConst * y <= 3);

  solver->setMaximizeObjective(x);
  solver->optimize();

  auto xVal = solver->getValue(x).value();
  auto yVal = solver->getValue(y).value();
  auto objVal = solver->getObjective().value();

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
  EXPECT_NEAR(a.get(GRB_DoubleAttr_X), xVal, 1e-6);
  EXPECT_NEAR(b.get(GRB_DoubleAttr_X), yVal, 1e-6);
  EXPECT_NEAR(model.get(GRB_DoubleAttr_ObjVal), objVal, 1e-6);
}

// [END AI-generated test cases]

// Factories for both solvers
std::unique_ptr<CPSolver> makeCbc() { return std::make_unique<CbcSolver>(); }

std::unique_ptr<CPSolver> makeGurobi() {
  return std::make_unique<GurobiSolver>();
}

// Runs all MILP test with two different solvers
INSTANTIATE_TEST_SUITE_P(SolverImplementations, ParamSolverTest,
                         ::testing::Values(makeCbc, makeGurobi));

#endif
} // namespace
