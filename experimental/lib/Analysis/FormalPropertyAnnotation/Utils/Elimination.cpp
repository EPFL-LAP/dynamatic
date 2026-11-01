//===- Elimination.cpp - Elimination of variables from equalities *- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the elimination of variables with the Presburger library of MLIR.
//
//===----------------------------------------------------------------------===//

#include "Elimination.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "llvm/Support/ErrorHandling.h"

using namespace dynamatic;
using namespace mlir::presburger;

std::vector<SmallVector<int64_t>>
dynamatic::eliminateVariables(ArrayRef<SmallVector<int64_t>> equalities,
                              unsigned numVars, unsigned numEliminated) {
  assert(numEliminated <= numVars && "more variables to eliminate than exist");
  IntegerRelation system(PresburgerSpace::getSetSpace(numVars));
  for (const SmallVector<int64_t> &equality : equalities) {
    assert(equality.size() == numVars && "one coefficient per variable");
    // An equality of an IntegerRelation has `numVars + 1` entries: one
    // coefficient per variable, followed by a constant term (c_0 * x_0 + ... +
    // c_n-1 * x_n-1 + c = 0). The equalities here have only the coefficients,
    // so the constant term c = 0 is appended.
    // TODO: Take the constant term from the equality (see the header).
    SmallVector<int64_t> withConstant(equality);
    withConstant.push_back(0);
    system.addEquality(withConstant);
  }
  // With equalities only, the projection substitutes each eliminated variable
  // in all equalities (Gaussian elimination), and removes it with the equality
  // that defined it
  system.projectOut(0, numEliminated);
  assert(system.getNumVars() == numVars - numEliminated &&
         "a variable could not be eliminated");
  // The remaining equalities may depend on each other: the row echelon form
  // drops the dependent ones (and the duplicates)
  system.simplify();

  std::vector<SmallVector<int64_t>> implied;
  for (unsigned row = 0, e = system.getNumEqualities(); row < e; ++row) {
    SmallVector<int64_t> coefficients;
    for (unsigned col = 0, n = system.getNumVars(); col < n; ++col) {
      // The elimination uses arbitrary-precision integers, so that the
      // coefficients cannot overflow before this point
      const DynamicAPInt &coef = system.atEq(row, col);
      if (coef < std::numeric_limits<int64_t>::min() ||
          coef > std::numeric_limits<int64_t>::max())
        llvm::report_fatal_error("coefficient of an implied equality does not "
                                 "fit in 64 bits");
      coefficients.push_back(int64fromDynamicAPInt(coef));
    }
    implied.push_back(std::move(coefficients));
  }
  return implied;
}
