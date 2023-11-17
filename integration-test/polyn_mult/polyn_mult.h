//===- polyn_mult.h - Polynomial muliplication  -------------------*- C -*-===//
//
// Declares the polyn_mult kernel which computes a polynomial multiplication.
//
//===----------------------------------------------------------------------===//

#ifndef POLYN_MULT_POLYN_MULT_H
#define POLYN_MULT_POLYN_MULT_H

#include <stdint.h>

#define N 100

typedef uint32_t out_int_t;
typedef uint32_t in_int_t;

/// Computes the polynomial multiplication between the two first arrays and sets
/// the result in the last array.
unsigned polyn_mult(in_int_t a[N], in_int_t b[N], out_int_t out[N]);

#endif // POLYN_MULT_POLYN_MULT_H