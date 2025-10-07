#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include <gtest/gtest.h>

using namespace dynamatic::cp;

namespace {
// Tests factorial of negative numbers.
TEST(BasicTests, test1) {

  // [Using our API to solve the result]
  auto solver = GurobiSolver();
  auto x = solver.addVariable("x", Var::REAL, /* lb */ 0, std::nullopt);
  auto y = solver.addVariable("x", Var::REAL, /* lb */ 0, std::nullopt);
  solver.addLinearConstraint(x + 2 * y <= 14);
  solver.addLinearConstraint(3 * x - y >= 0);
  solver.addLinearConstraint(x - y <= 2);
  solver.setMaximizeObjective(3 * x + 4 * y);
  solver.solve();
  auto xValue = solver.getValue(x);
  std::cerr << "Optimal value for x: " << xValue << "\n";
  auto yValue = solver.getValue(x);
  std::cerr << "Optimal value for y: " << yValue << "\n";

  // [Using Gurobi's API to solve the result]

  GRBEnv env = GRBEnv(true);
  env.start();

  // Create an empty model
  GRBModel model = GRBModel(env);

  // Create variables x and y (continuous by default)
  GRBVar a =
      model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x");
  GRBVar b =
      model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "y");

  // Add constraints
  model.addConstr(a + 2 * b <= 14, "c1");
  model.addConstr(3 * a - b >= 0, "c2");
  model.addConstr(a - b <= 2, "c3");

  // Set objective: maximize 3*x + 4*y
  model.setObjective(3 * a + 4 * b, GRB_MAXIMIZE);

  // Optimize the model
  model.optimize();

  // EXPECT_EQ(model.get(GRB_DoubleAttr_ObjVal), solver.getValue(x));
  EXPECT_EQ(a.get(GRB_DoubleAttr_X), solver.getValue(x));
  EXPECT_EQ(b.get(GRB_DoubleAttr_X), solver.getValue(y));

  // Output results
  std::cout << "Optimal x: " << a.get(GRB_DoubleAttr_X) << std::endl;
  std::cout << "Optimal y: " << b.get(GRB_DoubleAttr_X) << std::endl;
  std::cout << "Optimal objective: " << model.get(GRB_DoubleAttr_ObjVal)
            << std::endl;
}
} // namespace
