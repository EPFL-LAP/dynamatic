//===- kernel_2mm_float.c -----------------------------------------*- C -*-===//
//
// kernel_2mm_float.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include <stdlib.h>

#define NI 10
#define NJ 10
#define NK 10
#define NL 10
#define N 10

void kernel_2mm_float(float alpha, float beta, float tmp[N][N], float A[N][N],
                      float B[N][N], float C[N][N], float D[N][N]) {
  int i, j, k;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      float x = 0.0;
      for (k = 0; k < NK; ++k)
        x += alpha * A[i][k] * B[k][j];
      tmp[i][j] = x;
    }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      float x = D[i][j] * beta;
      for (k = 0; k < NJ; ++k)
        x += tmp[i][k] * C[k][j];
      D[i][j] = x;
    }
}

int main(void) {
  float alpha;
  float beta;
  float tmp[N][N];
  float A[N][N];
  float B[N][N];
  float C[N][N];
  float D[N][N];

  alpha = 1;
  beta = 1;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i][j] = rand() % 2;
      B[i][j] = rand() % 2;
      C[i][j] = rand() % 2;
      D[i][j] = rand() % 2;
    }
  }

  CALL_KERNEL(kernel_2mm_float, alpha, beta, tmp, A, B, C, D);
  return 0;
}
