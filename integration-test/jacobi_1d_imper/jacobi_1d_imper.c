//===- jacobi_1d_imper.c ------------------------------------------*- C -*-===//
//
// jacobi_1d_imper.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "jacobi_1d_imper.h"
#include <stdlib.h>

void jacobi_1d_imper(inout_int_t A[N], inout_int_t B[N]) {
  int t, i, j;

  for (t = 0; t < TSTEPS; t++) {
    for (i = 1; i < N - 1; i++)
      B[i] = 3 * (A[i - 1] + A[i] + A[i + 1]);
    for (j = 1; j < N - 1; j++)
      A[j] = B[j];
  }
}

int main(void) {

  inout_int_t A[N];
  inout_int_t B[N];

  for (int i = 0; i < N; ++i) {
    A[i] = rand() % 100;
    B[i] = rand() % 100;
  }

  CALL_KERNEL(jacobi_1d_imper, A, B);
  return 0;
}
