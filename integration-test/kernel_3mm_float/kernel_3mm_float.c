//===- kernel_3mm_float.c -----------------------------------------*- C -*-===//
//
// kernel_3mm_float.c: This file adapted from the PolyBench/C 3.2 test suite.
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
#define NM 10
#define N 10

void kernel_3mm_float(float A[N][N], float B[N][N], float C[N][N],
                      float D[N][N], float E[N][N], float F[N][N],
                      float G[N][N]) {
  int i, j, k;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      float tmp = E[i][j];
      for (k = 0; k < NK; ++k)
        tmp += A[i][k] * B[k][j];
      E[i][j] = tmp;
    }

  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++) {
      float tmp = F[i][j];
      for (k = 0; k < NM; ++k)
        tmp += C[i][k] * D[k][j];
      F[i][j] = tmp;
    }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      float tmp = G[i][j];
      for (k = 0; k < NJ; ++k)
        tmp += E[i][k] * F[k][j];
      G[i][j] = tmp;
    }
}

int main(void) {
  float A[N][N];
  float B[N][N];
  float C[N][N];
  float D[N][N];
  float E[N][N];
  float F[N][N];
  float G[N][N];

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i][j] = rand() % 2;
      B[i][j] = rand() % 2;
      C[i][j] = rand() % 2;
      D[i][j] = rand() % 2;
      E[i][j] = rand() % 2;
      F[i][j] = rand() % 2;
      G[i][j] = rand() % 2;
    }
  }

  CALL_KERNEL(kernel_3mm_float, A, B, C, D, E, F, G);
  return 0;
}
