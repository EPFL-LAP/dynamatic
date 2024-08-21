//===- trisolv.c --------------------------------------------------*- C -*-===//
//
// trisolv.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "trisolv.h"
#include <stdlib.h>

void trisolv(out_int_t x[N], in_int_t A[N][N], in_int_t c[N]) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = c[i];
    for (j = 0; j <= i - 1; j++)
      x[i] = x[i] - A[i][j] * x[j];
    x[i] = x[i] / A[i][i];
  }
}

int main(void) {
  out_int_t xArray[N];
  in_int_t A[N][N];
  in_int_t c[N];

  for (int i = 0; i < N; ++i) {
    c[i] = rand() % 100;
    for (int j = 0; j < N; ++j) {
      A[i][j] = rand() % 100 + 1;
    }
  }

  CALL_KERNEL(trisolv, xArray, A, c);
}
