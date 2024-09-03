//===- correlation_float.c ----------------------------------------*- C -*-===//
//
// correlation_float.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "correlation_float.h"
#include "dynamatic/Integration.h"
#include <math.h>
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

void correlation_float(inout_float_t data[30][30], inout_float_t mean[30],
                       out_float_t symmat[30][30], inout_float_t stddev[30],
                       in_float_t float_n) {
  int i, j, j1, j2;
  float eps = 0.1f;

  /* Determine mean of column vectors of input data matrix */
  for (j = 0; j < 10; j++) {
    float x = 0.0;
    for (i = 0; i < 10; i++)
      x += data[i][j];
    mean[j] = x / float_n;
  }

  /* Determine standard deviations of column vectors of data matrix. */
  for (j = 0; j < 10; j++) {
    float x = 0.0;
    for (i = 0; i < 10; i++)
      x += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    x /= float_n;
    x = sqrt(stddev[j]);
    /* The following in an inelegant but usual way to handle
       near-zero std. dev. values, which below would cause a zero- divide. */
    stddev[j] = x <= eps ? 1.0 : x;
  }

  /* Center and reduce the column vectors. */
  for (i = 0; i < 10; i++)
    for (j = 0; j < 10; j++) {
      float x = data[i][j];
      x -= mean[j];
      x /= sqrt(float_n) * stddev[j];
      data[i][j] = x;
    }

  /* Calculate the m * m correlation matrix. */
  for (j1 = 0; j1 < 9; j1++) {
    symmat[j1][j1] = 1.0;
    for (j2 = j1 + 1; j2 < 10; j2++) {
      float x = 0.0;
      for (i = 0; i < 10; i++)
        x += (data[i][j1] * data[i][j2]);
      symmat[j1][j2] = x;
      symmat[j2][j1] = x;
    }
  }
  symmat[9][9] = 1.0;
}

int main(void) {
  inout_float_t data[30][30];
  out_float_t symmat[30][30];
  inout_float_t mean[30];
  inout_float_t stddev[30];
  in_float_t float_n;

  for (int i = 0; i < 30; ++i) {
    for (int j = 0; j < 30; ++j) {
      data[i][j] = rand() % 100;
    }
  }
  float_n = rand() % 100;

  CALL_KERNEL(correlation_float, data, mean, symmat, stddev, float_n);
}
