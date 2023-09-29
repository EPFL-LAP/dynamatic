//===- polyn_mult.h - Polynomial muliplication  -------------------*- C -*-===//
//
// Declares the polyn_mult kernel which computes a polynomial multiplication.
//
//===----------------------------------------------------------------------===//

#ifndef POLYN_MULT_POLYN_MULT_H
#define POLYN_MULT_POLYN_MULT_H

#include <cstddef>
#include <stdint.h>

#define N 100

/// Computes the polynomial multiplication between the two first arrays and sets
/// the result in the last array.
// NOLINTNEXTLINE(readability-identifier-naming)
size_t polyn_mult(uint32_t a[N], uint32_t b[N], uint32_t out[N]);

#endif // POLYN_MULT_POLYN_MULT_H