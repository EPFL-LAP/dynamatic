//===- gaussian.c - Gaussian computation --------------------------*- C -*-===//
//
// Implements the gaussian kernel.
//
//===----------------------------------------------------------------------===//

#include "gaussian.h"

size_t gaussian(int c[N], int a[N][N]) {
  size_t sum = 0;
  for (size_t j = 1; j <= N_DEC_DEC; j++) {
    for (size_t i = j + 1; i <= N_DEC_DEC; i++) {
      for (size_t k = 1; k <= N_DEC; k++) {
        a[i][k] = a[i][k] - c[j] * a[j][k];
        sum += k;
      }
    }
  }
  return sum;
}
