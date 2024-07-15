//===- gesummv_float.c ----------------------------------------- -*- C -*-===//
//
// This file is adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//
#include "gesummv_float.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void gesummv_float(in_float_t alpha, in_float_t beta, in_float_t A[30][30],
                   in_float_t B[30][30], out_float_t tmp[30], out_float_t y[30],
                   in_float_t x[30]) {
  int i, j, k;

  for (i = 0; i < 30; i++) {
    float t_tmp = 0;
    float t_y = 0;

    for (j = 0; j < 30; j++) {
      float t_x = x[j];
      t_tmp = A[i][j] * t_x + t_tmp;
      t_y = B[i][j] * t_x + t_y;
    }

    tmp[i] = t_tmp;
    y[i] = alpha * t_tmp + beta * t_y;
  }
}

int main(void) {
  in_float_t alpha;
  in_float_t beta;
  in_float_t A[30][30];
  in_float_t B[30][30];
  out_float_t tmp[30];
  out_float_t y[30];
  in_float_t x[30];

  alpha = 1;
  beta = 1;
  for (int j = 0; j < 30; ++j) {
    tmp[j] = 1;
    x[j] = rand() % 2;
    y[j] = rand() % 2;
    for (int k = 0; k < 30; ++k) {
      A[j][k] = rand() % 3;
      B[j][k] = rand() % 2;
    }
  }

  CALL_KERNEL(gesummv_float, alpha, beta, A, B, tmp, y, x);
  return 0;
}
