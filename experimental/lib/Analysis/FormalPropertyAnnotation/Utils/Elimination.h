//===- Elimination.h - Elimination of variables from equalities -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Eliminates variables from a system of homogeneous linear equalities with
/// integer coefficients, e.g., the flow variables that cannot be observed from
/// the flow equations of a circuit.
///
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_FORMALPROPERTYANNOTATION_UTILS_ELIMINATION_H
#define EXPERIMENTAL_ANALYSIS_FORMALPROPERTYANNOTATION_UTILS_ELIMINATION_H

#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <vector>

namespace dynamatic {

/// The equalities over the last `numVars - numEliminated` variables that
/// `equalities` imply, i.e., without the first `numEliminated` variables. Each
/// of the `equalities` has `numVars` coefficients, one per variable, and states
/// that the weighted sum of the variables is 0. The result is a basis of the
/// implied equalities in row echelon form: each row has `numVars -
/// numEliminated` coefficients, divided by their greatest common divisor, e.g.,
/// for the variables (l, x, y), with `numVars` = 3 and `numEliminated` = 1:
///   equalities: {1, -1, 0}, {1, 0, -1}   (l - x = 0, l - y = 0)
///   result:     {1, -1}                  (x - y = 0, up to its sign)
/// Reports a fatal error if a coefficient of the result does not fit in 64
/// bits.
///
/// TODO: Support equalities with a constant term (e.g., the token of an
/// argument of the function, which is pending at reset: lambda_arg + valid =
/// 1), as rows of `numVars + 1` entries with the constant term last, both for
/// `equalities` and for the result.
std::vector<SmallVector<int64_t>>
eliminateVariables(ArrayRef<SmallVector<int64_t>> equalities, unsigned numVars,
                   unsigned numEliminated);

} // namespace dynamatic

#endif // EXPERIMENTAL_ANALYSIS_FORMALPROPERTYANNOTATION_UTILS_ELIMINATION_H
