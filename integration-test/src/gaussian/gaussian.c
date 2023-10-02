//===- gaussian.c - Gaussian computation --------------------------*- C -*-===//
//
// Implements the gaussian kernel.
//
//===----------------------------------------------------------------------===//

#include "gaussian.h"
#include "stdlib.h"

unsigned gaussian(in_int_t c[N], inout_int_t a[N][N]) {
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

int main(void) {
  in_int_t c[20];
  in_int_t a[20][20];

  srand(13);
  for (int y = 0; y < 20; ++y) {
    c[y] = 1;
    for (int x = 0; x < 20; ++x) {
      a[y][x] = 1;
    }
  }

  gaussian(c, a);
  return 0;
}
