//===- bicg_float.c ---------------------------------------------*- C -*-===//
//
// bicg_float.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===-------------------------------------------------------------------===//

#include "bicg_float.h"

#include "dynamatic/Integration.h"
#include <stdlib.h>

float bicg_float(in_float_t A[N][N], inout_float_t s[N], inout_float_t q[N],
                 in_float_t p[N], in_float_t r[N]) {
  int i, j;

  float tmp_q = 0;

  for (i = 0; i < N; i++) {
    tmp_q = q[i];
    for (j = 0; j < N; j++) {
      float tmp = A[i][j];
      s[j] = s[j] + r[i] * tmp;
      tmp_q = tmp_q + tmp * p[j];
    }
    q[i] = tmp_q;
  }
  return tmp_q;
}

int main(void) {
  in_float_t A[N][N];
  inout_float_t s[N];
  inout_float_t q[N];
  in_float_t p[N];
  in_float_t r[N];

  for (int i = 0; i < N; ++i) {
    s[i] = rand() % 100;
    q[i] = rand() % 100;
    p[i] = rand() % 100;
    r[i] = rand() % 100;
    for (int j = 0; j < N; ++j) {
      A[i][j] = rand() % 100;
    }
  }

  CALL_KERNEL(bicg_float, A, s, q, p, r);
  return 0;
}
