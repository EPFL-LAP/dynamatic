//===- matvec.c - Multiply matrix to vector  ----------------------*- C -*-===//
//
// Implements the matvec kernel.
//
//===----------------------------------------------------------------------===//

#include "matvec.h"

int matvec(int m[N][N], int v[N], int out[N]) {
  int tmp = 0;
  for (unsigned i = 0; i < N; i++) {
    tmp = 0;
    for (unsigned j = 0; j < N; j++) {
      tmp += v[j] * m[i][j];
    }
    out[i] = tmp;
  }
  return tmp;
}
