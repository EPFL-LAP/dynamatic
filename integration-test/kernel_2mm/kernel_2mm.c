/**
 * 2mm.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */

#include "kernel_2mm.h"
#include "dynamatic/Integration.h"
#include <stdio.h>
#include <stdlib.h>

int kernel_2mm(in_int_t alpha, in_int_t beta, inout_int_t tmp[NI][NJ],
               in_int_t A[NI][NK], in_int_t B[NK][NJ], in_int_t C[NK][NL],
               inout_int_t D[NI][NL]) {
  int ii;
  for (unsigned i = 0; i < NI; i++) {
    for (unsigned j = 0; j < NJ; j++) {
      tmp[i][j] = 0;
      for (unsigned k = 0; k < NK; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
    }
  }

  for (ii = 0; ii < NI; ii++) {
    for (unsigned l = 0; l < NL; l++) {
      D[ii][l] *= beta;
      for (unsigned k = 0; k < NJ; ++k)
        D[ii][l] += tmp[ii][k] * C[k][l];
    }
  }

  return ii;
}

int main(void) {
  in_int_t alpha;
  in_int_t beta;
  inout_int_t tmp[NI][NJ];
  in_int_t A[NI][NK];
  in_int_t B[NK][NJ];
  in_int_t C[NK][NL];
  inout_int_t D[NI][NL];

  alpha = rand();
  beta = rand();
  for (unsigned i = 0; i < NI; ++i) {
    for (unsigned k = 0; k < NK; ++k)
      A[i][k] = rand() % 100;
    for (unsigned l = 0; l < NL; ++l)
      D[i][l] = rand() % 100;
  }

  for (unsigned k = 0; k < NK; ++k) {
    for (unsigned j = 0; j < NJ; ++j)
      B[k][j] = rand() % 100;
    for (unsigned l = 0; l < NL; ++l)
      C[k][l] = rand() % 100;
  }

  CALL_KERNEL(kernel_2mm, alpha, beta, tmp, A, B, C, D);
  return 0;
}
