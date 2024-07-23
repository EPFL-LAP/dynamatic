
//===- gemver_float.c ------------------------------------------- -*- C -*-===//
//
// This file is adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "gemver_float.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void gemver_float(in_float_t alpha, in_float_t beta, inout_float_t A[N][N],
                  in_float_t u1[N], in_float_t v1[N], in_float_t u2[N],
                  in_float_t v2[N], inout_float_t w[N], inout_float_t x[N],
                  in_float_t y[N], in_float_t z[N]) {
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < N; i++) {
    float tmp = x[i];
    for (j = 0; j < N; j++)
      tmp = tmp + beta * A[j][i] * y[j];
    x[i] = tmp;
  }

  for (i = 0; i < N; i++)

    x[i] = x[i] + z[i];

  for (i = 0; i < N; i++) {
    float tmp = w[i];
    for (j = 0; j < N; j++)
      tmp = tmp + alpha * A[i][j] * x[j];
    w[i] = tmp;
  }
}

int main(void) {
  in_float_t alpha;
  in_float_t beta;
  inout_float_t A[N][N];
  in_float_t u1[N];
  in_float_t v1[N];
  in_float_t u2[N];
  in_float_t v2[N];
  inout_float_t w[N];
  inout_float_t x[N];
  in_float_t y[N];
  in_float_t z[N];

  alpha = rand() % 20;
  beta = rand() % 20;
  for (int yy = 0; yy < N; ++yy) {
    u1[yy] = rand() % 20;
    v1[yy] = rand() % 20;
    u2[yy] = rand() % 20;
    v2[yy] = rand() % 20;
    w[yy] = rand() % 20;
    x[yy] = rand() % 20;
    y[yy] = rand() % 20;
    z[yy] = rand() % 20;
    for (int x = 0; x < N; ++x) {
      A[yy][x] = rand() % 10;
    }
  }

  CALL_KERNEL(gemver_float, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
  return 0;
}
