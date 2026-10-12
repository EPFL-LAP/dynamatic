//===- gemver_float.c ------------------------------------------- -*- C -*-===//
//
// This file is adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include <stdlib.h>

#define N 10

void gemver_float(float alpha, float beta, float A[N][N], float u1[N],
                  float v1[N], float u2[N], float v2[N], float w[N], float x[N],
                  float y[N], float z[N]) {
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
  float alpha;
  float beta;
  float A[N][N];
  float u1[N];
  float v1[N];
  float u2[N];
  float v2[N];
  float w[N];
  float x[N];
  float y[N];
  float z[N];

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
    for (int xx = 0; xx < N; ++xx) {
      A[yy][xx] = rand() % 10;
    }
  }

  CALL_KERNEL(gemver_float, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
  return 0;
}
