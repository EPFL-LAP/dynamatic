//===- matvec.c - Multiply matrix to vector  ----------------------*- C -*-===//
//
// Implements the matvec kernel.
//
//===----------------------------------------------------------------------===//

#include "matvec.h"
#include <cstddef>

int matvec(int m[N][N], int v[N], int out[N]) {
  int tmp = 0;
  for (size_t i = 0; i < N; i++) {
    tmp = 0;
    for (size_t j = 0; j < N; j++) {
      tmp += v[j] * m[i][j];
    }
    out[i] = tmp;
  }
  return tmp;
}
