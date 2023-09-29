//===- polyn_mult.c - Polynomial muliplication  -------------------*- C -*-===//
//
// Implements the polyn_mult kernel.
//
//===----------------------------------------------------------------------===//

#include "polyn_mult.h"

size_t polyn_mult(uint32_t a[N], uint32_t b[N], uint32_t out[N]) {
  size_t p = 0;
  for (size_t k = 0; k < N; k++) {
    out[k] = 0;
    size_t i = 0;
    for (i = 1; i < N - k; i++)
      out[k] += a[k + i] * b[N - i];
    for (i = 0; i < k + 1; i++)
      out[k] += a[k - i] * b[i];
    p = i + k;
  }
  return p;
}
