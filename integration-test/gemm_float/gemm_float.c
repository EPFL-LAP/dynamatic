//===- gemm_float.c ---------------------------------------------*- C -*-===//
//
// gemm_float.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===-------------------------------------------------------------------===//

#include "gemm_float.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>
#define N 4000

void gemm_float(in_float_t alpha, in_float_t beta, in_float_t A[30][30],
                in_float_t B[30][30], inout_float_t C[30][30]) {
  int i, j, k;

  for (i = 0; i < 20; i++)
    for (j = 0; j < 20; j++) {
      float tmp = C[i][j] * beta;
      for (k = 0; k < 20; ++k)
        tmp += alpha * A[i][k] * B[k][j];
      C[i][j] = tmp;
    }
}

int main(void) {
  in_float_t alpha;
  in_float_t beta;
  in_float_t A[30][30];
  in_float_t B[30][30];
  out_float_t C[30][30];

  alpha = 1;
  beta = 1;
  for (int i = 0; i < 30; ++i) {
    for (int j = 0; j < 30; ++j) {
      A[i][j] = rand() % 2;
      B[i][j] = rand() % 3;
      C[i][j] = 1;
    }
  }

  CALL_KERNEL(gemm_float, alpha, beta, A, B, C);
  return 0;
}
