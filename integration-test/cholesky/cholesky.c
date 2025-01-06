//===- cholesky.c -----------------------------------------------*- C -*-===//
//
//===-------------------------------------------------------------------===//
//
#include "cholesky.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void cholesky(inout_float_t A[N][N], out_float_t R[N][N]) {
  for (int k = 0; k < N; k++) {
    R[k][k] = A[k][k]; // sqrt(A[k][k]);
    for (int j = k + 1; j < N; j++) {
      R[k][j] = A[k][j] / R[k][k];
    }
    for (int j = k + 1; j < N; j++) {
      for (int i = j; i < N; i++) {
        A[j][i] = A[j][i] - R[k][j] * R[k][i];
      }
    }
  }
}

int main(void) {
  inout_float_t A[N][N];
  out_float_t R[N][N];

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j)
      A[i][j] = rand() % 10 + 10;
  }

  CALL_KERNEL(cholesky, A, R);
  return 0;
}
