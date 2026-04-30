//===- kernel_2mm.c -----------------------------------------------*- C -*-===//
//
// kernel_2mm_float.c: This file adapted from the PolyBench/C 3.2 test suite.
//
// Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
// Web address: http://polybench.sourceforge.net
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include <stdio.h>
#include <stdlib.h>

#define NI 10
#define NJ 10
#define NK 10
#define NL 10

void kernel_2mm(int alpha, int beta, int tmp[NI][NJ], int A[NI][NK],
                int B[NK][NJ], int C[NK][NL], int D[NI][NL]) {
  for (unsigned i = 0; i < NI; i++) {
    for (unsigned j = 0; j < NJ; j++) {
      int acc = 0;
      for (unsigned k = 0; k < NK; ++k)
        acc += alpha * A[i][k] * B[k][j];
      tmp[i][j] = acc;
    }
  }

  for (unsigned i = 0; i < NI; i++) {
    for (unsigned l = 0; l < NL; l++) {
      int acc = D[i][l] * beta;
      for (unsigned k = 0; k < NJ; ++k)
        acc += tmp[i][k] * C[k][l];
      D[i][l] = acc;
    }
  }
}

int main(void) {
  int alpha;
  int beta;
  int tmp[NI][NJ];
  int A[NI][NK];
  int B[NK][NJ];
  int C[NK][NL];
  int D[NI][NL];

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
