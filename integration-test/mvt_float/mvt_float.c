//===- mvt_float.c ------------------------------------------------*- C -*-===//
//
// This file is part of the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Integration.h"
#include "mvt_float.h"
#include <stdlib.h>

#define N 4000

void mvt_float(in_float_t A[30][30], inout_float_t x1[30], inout_float_t x2[30],
               in_float_t y1[30], in_float_t y2[30]) {
  int i, j, k;

  for (i = 0; i < 30; i++) {
    float tmp = x1[i];
    for (j = 0; j < 30; j++)
      tmp = tmp + A[i][j] * y1[j];
    x1[i] = tmp;
  }

  for (i = 0; i < 30; i++) {
    float tmp = x2[i];
    for (j = 0; j < 30; j++)
      tmp = tmp + A[j][i] * y2[j];
    x2[i] = tmp;
  }
}

int main(void) {

  in_float_t A[30][30];
  inout_float_t x1[30];
  inout_float_t x2[30];
  in_float_t y1[30];
  in_float_t y2[30];

  for (int i = 0; i < 30; ++i) {
    x1[i] = rand() % 2;
    x2[i] = rand() % 2;
    y1[i] = rand() % 2;
    y2[i] = rand() % 2;
    for (int j = 0; j < 30; ++j) {
      A[i][j] = rand() % 2;
    }
  }

  CALL_KERNEL(mvt_float, A, x1, x2, y1, y2);
  return 0;
}
