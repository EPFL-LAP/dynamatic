//===- gemver.c - Vector multiplication and matrix addition -------*- C -*-===//
//
// Implements the gemver kernel.
//
// This file is partially taken from the Polybench 4.2.1 testsuite.
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/tree/master
//
//===----------------------------------------------------------------------===//

#include "gemver.h"
#include "../integration_utils.h"
#include <stdlib.h>

void gemver(in_int_t alpha, in_int_t beta, in_int_t u1[N], in_int_t v1[N],
            in_int_t u2[N], in_int_t v2[N], in_int_t y[N], in_int_t z[N],
            inout_int_t a[N][N], inout_int_t w[N], inout_int_t x[N]) {
  for (unsigned i = 0; i < N; i++)
    for (unsigned j = 0; j < N; j++)
      a[i][j] = a[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (unsigned i = 0; i < N; i++) {
    int tmp = x[i];
    for (unsigned j = 0; j < N; j++)
      tmp = tmp + beta * a[j][i] * y[j];
    x[i] = tmp;
  }

  for (unsigned i = 0; i < N; i++)
    x[i] = x[i] + z[i];

  for (unsigned i = 0; i < N; i++) {
    int tmp = w[i];
    for (unsigned j = 0; j < N; j++)
      tmp = tmp + alpha * a[i][j] * x[j];
    w[i] = tmp;
  }
}

int main(void) {
  in_int_t alpha;
  in_int_t beta;
  in_int_t u1[N];
  in_int_t v1[N];
  in_int_t u2[N];
  in_int_t v2[N];
  in_int_t y[N];
  in_int_t z[N];
  inout_int_t a[N][N];
  inout_int_t w[N];
  inout_int_t x[N];

  alpha = rand() % 20;
  beta = rand() % 20;
  for (unsigned yy = 0; yy < N; ++yy) {
    u1[yy] = rand() % 20;
    v1[yy] = rand() % 20;
    u2[yy] = rand() % 20;
    v2[yy] = rand() % 20;
    w[yy] = rand() % 20;
    x[yy] = rand() % 20;
    y[yy] = rand() % 20;
    z[yy] = rand() % 20;
    for (unsigned x = 0; x < N; ++x)
      a[yy][x] = rand() % 10;
  }

  CALL_KERNEL(gemver, alpha, beta, u1, v1, u2, v2, y, z, a, w, x);
  return 0;
}
