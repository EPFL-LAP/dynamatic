//===- symm_float.c -----------------------------------------------*- C -*-===//
//
// symm_float.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "symm_float.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

#define N 4000

void symm_float(in_float_t A[30][30], in_float_t B[30][30],
                inout_float_t C[30][30], in_float_t alpha, in_float_t beta) {
  int i, j, k;

  for (i = 0; i < 10; i++)
    for (j = 0; j < 30; j++) {
      float acc = 0;
      for (k = 0; k < j - 1; k++) {
        C[k][j] += alpha * A[k][i] * B[i][j];
        acc += B[k][j] * A[k][i];
      }
      C[i][j] = beta * C[i][j] + alpha * A[i][i] * B[i][j] + alpha * acc;
    }
}

int main(void) {

  in_float_t A[30][30];
  in_float_t B[30][30];
  inout_float_t C[30][30];
  in_float_t alpha;
  in_float_t beta;

  alpha = rand() % 2;
  beta = rand() % 2;

  for (int i = 0; i < 30; ++i) {
    for (int j = 0; j < 30; ++j) {
      A[i][j] = rand() % 2;
      B[i][j] = rand() % 2;
      C[i][j] = rand() % 2;
    }
  }

  CALL_KERNEL(symm_float, A, B, C, alpha, beta);
  return 0;
}
