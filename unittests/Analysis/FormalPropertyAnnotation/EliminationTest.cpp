//===- EliminationTest.cpp - Tests of the elimination of variables --------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests the elimination of variables from homogeneous linear equalities, which
// derives the flow invariants over the internal states of a circuit.
//
//===----------------------------------------------------------------------===//

#include "Utils/Elimination.h"
#include <gtest/gtest.h>
#include <vector>

using namespace dynamatic;

namespace {
using Rows = std::vector<SmallVector<int64_t>>;

/// The row with its sign chosen so that its first non-zero coefficient is
/// positive, as an equality is the same up to its sign.
SmallVector<int64_t> withPositiveLead(SmallVector<int64_t> row) {
  for (int64_t coef : row) {
    if (coef == 0)
      continue;
    if (coef < 0)
      for (int64_t &c : row)
        c = -c;
    break;
  }
  return row;
}

/// The column of the first non-zero coefficient of the row.
size_t leadingColumn(const SmallVector<int64_t> &row) {
  for (size_t col = 0; col < row.size(); ++col)
    if (row[col] != 0)
      return col;
  return row.size();
}
} // namespace

// Two paths from a fork reconverge at a join, one of them through a buffer:
// the sent states of the fork and the slot of the buffer are related, and the
// transfer counters (lambdas) are eliminated. Variables:
//   lambdaIn, lambdaOut0, lambdaOut1, lambdaBuf | sent0, sent1, full
TEST(EliminationTest, ReconvergentPaths) {
  Rows equations = {
      // Fork: lambdaIn + sent0 = lambdaOut0, lambdaIn + sent1 = lambdaOut1
      {1, -1, 0, 0, 1, 0, 0},
      {1, 0, -1, 0, 0, 1, 0},
      // Buffer: lambdaOut0 = lambdaBuf + full
      {0, 1, 0, -1, 0, 0, -1},
      // Join: lambdaBuf = lambdaOut1
      {0, 0, -1, 1, 0, 0, 0},
  };
  Rows implied = eliminateVariables(equations, /*numVars=*/7,
                                    /*numEliminated=*/4);
  // sent0 = sent1 + full
  ASSERT_EQ(implied.size(), 1u);
  EXPECT_EQ(withPositiveLead(implied[0]), (SmallVector<int64_t>{1, -1, -1}));
}

// A chain of units relates each slot only to transfer counters: nothing is
// implied over the slots alone. Variables: lambdaA, lambdaB | full
TEST(EliminationTest, NothingImplied) {
  Rows equations = {{1, -1, -1}};
  EXPECT_TRUE(eliminateVariables(equations, /*numVars=*/3,
                                 /*numEliminated=*/2)
                  .empty());
}

// Equalities that depend on others are dropped, and each row is divided by the
// greatest common divisor of its coefficients. Variables: x, y
TEST(EliminationTest, DependentEqualitiesDropped) {
  Rows equations = {{2, -2}, {1, -1}, {3, -3}};
  Rows implied = eliminateVariables(equations, /*numVars=*/2,
                                    /*numEliminated=*/0);
  ASSERT_EQ(implied.size(), 1u);
  EXPECT_EQ(withPositiveLead(implied[0]), (SmallVector<int64_t>{1, -1}));
}

// Large coefficients do not overflow: eliminating the variable multiplies the
// rows by 100000, whose square does not fit in 32 bits. Variables: lambda | x,
// y
TEST(EliminationTest, LargeCoefficients) {
  Rows equations = {{100000, -1, 0}, {100000, 0, -1}};
  Rows implied = eliminateVariables(equations, /*numVars=*/3,
                                    /*numEliminated=*/1);
  // x = y
  ASSERT_EQ(implied.size(), 1u);
  EXPECT_EQ(withPositiveLead(implied[0]), (SmallVector<int64_t>{1, -1}));
}

// The implied equalities are in row echelon form: the first non-zero column of
// each row is after that of the previous row. Variables: lambda | x, y, z
TEST(EliminationTest, RowEchelonForm) {
  Rows equations = {
      // x = lambda, y = lambda, z = lambda
      {1, -1, 0, 0},
      {1, 0, -1, 0},
      {1, 0, 0, -1},
  };
  Rows implied = eliminateVariables(equations, /*numVars=*/4,
                                    /*numEliminated=*/1);
  // x = y = z: two independent equalities
  ASSERT_EQ(implied.size(), 2u);
  EXPECT_LT(leadingColumn(implied[0]), leadingColumn(implied[1]));
}
