//===- atax_float.c -----------------------------------------------*- C -*-===//
//
// atax_float.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "atax_float.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

#define NX 20
#define NY 20
#define N 20

void atax_float(in_float_t A[N][N], in_float_t x[N], inout_float_t y[N],
                inout_float_t tmp[N]) {
  int i, j;

  for (i = 0; i < NX; i++) {
    float t = tmp[i];
    for (j = 0; j < NY; j++)
      t = t + A[i][j] * x[j];
    for (j = 0; j < NY; j++)
      y[j] = y[j] + A[i][j] * t;
    tmp[i] = t;
  }
}

int main(void) {
  in_float_t A[N][N];
  in_float_t x[N];
  inout_float_t y[N];
  inout_float_t tmp[N];

  for (int i = 0; i < N; ++i) {
    x[i] = rand() % 100;
    y[i] = 0;
    tmp[i] = 0;
    for (int j = 0; j < N; ++j) {
      A[i][j] = rand() % 100;
    }
  }

  CALL_KERNEL(atax_float, A, x, y, tmp);
  return 0;
}