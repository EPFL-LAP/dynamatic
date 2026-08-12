/**
 * 3mm.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */

#include "dynamatic/Integration.h"
#include <stdio.h>
#include <stdlib.h>

#define NI 10
#define NJ 10
#define NK 10
#define NL 10
#define NM 10
#define N 10

void kernel_3mm(int A[NI][NK], int B[NK][NJ], int C[NJ][NM], int D[NM][NL],
                int E[NI][NJ], int F[NJ][NL], int G[NI][NL]) {
  for (unsigned i = 0; i < NI; i++) {
    for (unsigned j = 0; j < NJ; j++) {
      int tmp = 0;
      for (unsigned k = 0; k < NK; ++k)
        tmp += A[i][k] * B[k][j];
      E[i][j] = tmp;
    }
  }

  for (unsigned j = 0; j < NJ; j++) {
    for (unsigned l = 0; l < NL; l++) {
      int tmp = 0;
      for (unsigned m = 0; m < NM; ++m)
        tmp += C[j][m] * D[m][l];
      F[j][l] = tmp;
    }
  }

  for (unsigned i = 0; i < NI; i++) {
    for (unsigned l = 0; l < NL; l++) {
      int tmp = 0;
      for (unsigned j = 0; j < NJ; ++j)
        tmp += E[i][j] * F[j][l];
      G[i][l] = tmp;
    }
  }
}

int main(void) {
  int A[NI][NK];
  int B[NK][NJ];
  int C[NJ][NM];
  int D[NM][NL];
  int E[NI][NJ];
  int F[NJ][NL];
  int G[NI][NL];

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
