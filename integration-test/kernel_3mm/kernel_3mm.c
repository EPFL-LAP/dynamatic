/**
 * 3mm.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */

#include "kernel_3mm.h"
#include "../integration_utils.h"
#include <stdio.h>
#include <stdlib.h>

void kernel_3mm(in_int_t A[NI][NK], in_int_t B[NK][NJ], in_int_t C[NJ][NM],
                in_int_t D[NM][NL], inout_int_t E[NI][NJ],
                inout_int_t F[NJ][NL], inout_int_t G[NI][NL]) {
  for (unsigned i = 0; i < NI; i++) {
    for (unsigned j = 0; j < NJ; j++) {
      E[i][j] = 0;
      for (unsigned k = 0; k < NK; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }
  }

  for (unsigned j = 0; j < NJ; j++) {
    for (unsigned l = 0; l < NL; l++) {
      F[j][l] = 0;
      for (unsigned m = 0; m < NM; ++m)
        F[j][l] += C[j][m] * D[m][l];
    }
  }

  for (unsigned i = 0; i < NI; i++) {
    for (unsigned l = 0; l < NL; l++) {
      G[i][l] = 0;
      for (unsigned j = 0; j < NJ; ++j)
        G[i][l] += E[i][j] * F[j][l];
    }
  }
}

int main(void) {
  in_int_t A[NI][NK];
  in_int_t B[NK][NJ];
  in_int_t C[NJ][NM];
  in_int_t D[NM][NL];
  inout_int_t E[NI][NJ];
  inout_int_t F[NJ][NL];
  inout_int_t G[NI][NL];

  for (unsigned i = 0; i < NI; ++i) {
    for (unsigned k = 0; k < NK; ++k)
      A[i][k] = rand() % 10;
    for (unsigned j = 0; j < NJ; ++j)
      E[i][j] = rand() % 10;
    for (unsigned l = 0; l < NL; ++l)
      G[i][l] = rand() % 10;
  }

  for (unsigned j = 0; j < NJ; ++j) {
    for (unsigned m = 0; m < NM; ++m)
      C[j][m] = rand() % 10;
    for (unsigned l = 0; l < NL; ++l)
      F[j][l] = rand() % 10;
  }

  for (unsigned k = 0; k < NK; ++k) {
    for (unsigned j = 0; j < NJ; ++j)
      B[k][j] = rand() % 10;
  }

  for (unsigned m = 0; m < NM; ++m) {
    for (unsigned l = 0; l < NL; ++l)
      D[m][l] = rand() % 10;
  }

  CALL_KERNEL(kernel_3mm, A, B, C, D, E, F, G);
  return 0;
}
