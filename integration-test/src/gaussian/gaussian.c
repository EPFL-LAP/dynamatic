//===- gaussian.c - Gaussian computation --------------------------*- C -*-===//
//
// Implements the gaussian kernel.
//
//===----------------------------------------------------------------------===//

#include "gaussian.h"

unsigned gaussian(int c[N], int a[N][N]) {
  unsigned sum = 0;
  for (unsigned j = 1; j <= N_DEC_DEC; j++) {
    for (unsigned i = j + 1; i <= N_DEC_DEC; i++) {
      for (unsigned k = 1; k <= N_DEC; k++) {
        a[i][k] = a[i][k] - c[j] * a[j][k];
        sum += k;
      }
    }
  }
  return sum;
}
