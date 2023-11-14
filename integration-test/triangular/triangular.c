//===- triangular.c - Triangular multiplication -------------------*- C -*-===//
//
// Implements the triangular kernel.
//
//===----------------------------------------------------------------------===//

#include "triangular.h"
#include "../integration_utils.h"
#include <stdlib.h>

void triangular(in_int_t x[N], in_int_t n, inout_int_t a[N][N]) {
  for (int i = n - 1; i >= 0; i--)
    for (int k = i - 1; k >= 0; k--)
      a[k][n] -= a[k][i] * x[i];
}

int main(void) {
  in_int_t x[N];
  in_int_t n;
  inout_int_t a[N][N];

  n = N;
  for (int i = 0; i < N; ++i) {
    x[i] = rand() % 100;
    for (int y = 0; y < N; ++y)
      a[y][i] = rand() % 100;
  }

  CALL_KERNEL(triangular, x, n, a);
  return 0;
}
