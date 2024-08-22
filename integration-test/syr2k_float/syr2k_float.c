//===- syr2k_float.c ----------------------------------------------*- C -*-===//
//
// syr2k_float.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "syr2k_float.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>
#define N 4000

void syr2k_float(in_float_t A[30][30], in_float_t B[30][30],
                 inout_float_t C[30][30], in_float_t alpha, in_float_t beta) {
  int i, j, k;

  for (i = 0; i < 10; i++)
    for (j = 0; j < 10; j++)
      C[i][j] *= beta;
  for (i = 0; i < 10; i++)
    for (j = 0; j < 10; j++) {
      float tmp = C[i][j];
      for (k = 0; k < 10; k++) {
        tmp += alpha * A[i][k] * B[j][k];
        tmp += alpha * B[i][k] * A[j][k];
      }
      C[i][j] = tmp;
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

  CALL_KERNEL(syr2k_float, A, B, C, alpha, beta);
  return 0;
}
