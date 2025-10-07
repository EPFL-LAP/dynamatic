#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include <gtest/gtest.h>

using namespace dynamatic::cp;

namespace {
TEST(BasicTests, basicMILPTest) {

  // [Using our API to solve the result]
  auto solver = GurobiSolver();
  auto x = solver.addVariable("x", Var::REAL, /* lb */ 0, std::nullopt);
  auto y = solver.addVariable("y", Var::REAL, /* lb */ 0, std::nullopt);
  solver.addLinearConstraint(x + 2 * y <= 14);
  solver.addLinearConstraint(3 * x - y >= 0);
  solver.addLinearConstraint(x - y <= 2);
  solver.setMaximizeObjective(3 * x + 4 * y);
  solver.optimize();

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
  EXPECT_EQ(model.get(GRB_DoubleAttr_ObjVal), solver.getObjective());
  EXPECT_EQ(a.get(GRB_DoubleAttr_X), solver.getValue(x));
  EXPECT_EQ(b.get(GRB_DoubleAttr_X), solver.getValue(y));
}

TEST(BasicTests, basicIntegerProgramTest) {
  // [Using our API to solve the result]
  auto solver = GurobiSolver();
  auto x = solver.addVariable("x", Var::INTEGER, /* lb */ 0,
                              /* infinity */ std::nullopt);
  auto y = solver.addVariable("y", Var::INTEGER, /* lb */ 0,
                              /* infinity */ std::nullopt);
  auto z = solver.addVariable("z", Var::INTEGER, /* lb */ 0,
                              /* infinity */ std::nullopt);
  solver.addLinearConstraint(2 * x + 7 * y + 3 * z <= 50);
  solver.addLinearConstraint(3 * x + 5 * y + 7 * z <= 45);
  solver.addLinearConstraint(5 * x + 2 * y - 6 * z <= 37);
  solver.setMaximizeObjective(2 * x + 2 * y + 3 * z);
  solver.optimize();

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
  EXPECT_EQ(model.get(GRB_DoubleAttr_ObjVal), solver.getObjective());
  EXPECT_EQ(a.get(GRB_DoubleAttr_X), solver.getValue(x));
  EXPECT_EQ(b.get(GRB_DoubleAttr_X), solver.getValue(y));
  EXPECT_EQ(c.get(GRB_DoubleAttr_X), solver.getValue(z));
}

} // namespace
