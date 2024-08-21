//===- covariance.c -----------------------------------------------*- C -*-===//
//
// covariance.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "covariance.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

void covariance(inout_int_t data[30][30], out_int_t symmat[30][30],
                out_int_t mean[30]) {
  int i, j, j1, j2;
  int float_n = 30;

  /* Determine mean of column vectors of input data matrix */
  for (j = 0; j < 30; j++) {
    int x = 0;
    for (i = 0; i < 30; i++)
      x += data[i][j];
    mean[j] = x / float_n;
  }

  /* Center the column vectors. */
  for (i = 0; i < 30; i++)
    for (j = 0; j < 30; j++)
      data[i][j] -= mean[j];

  /* Calculate the m * m covariance matrix. */
  for (j1 = 0; j1 < 30; j1++)
    for (j2 = j1; j2 < 30; j2++) {
      int x = 0;
      for (i = 0; i < 30; i++)
        x += data[i][j1] * data[i][j2];

      symmat[j1][j2] = x;
      symmat[j2][j1] = x;
    }
}

int main(void) {
  inout_int_t data[30][30];
  out_int_t symmat[30][30];
  out_int_t mean[30];

  for (int i = 0; i < 30; ++i) {
    for (int j = 0; j < 30; ++j) {
      data[i][j] = rand() % 100;
    }
  }

  CALL_KERNEL(covariance, data, symmat, mean);
}
