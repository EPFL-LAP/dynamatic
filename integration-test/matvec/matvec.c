//===- matvec.c - Multiply matrix to vector  ----------------------*- C -*-===//
//
// Implements the matvec kernel.
//
//===----------------------------------------------------------------------===//

#include "matvec.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int matvec(in_int_t m[N][N], in_int_t v[N], out_int_t out[N]) {
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

int main(void) {
  in_int_t m[N][N];
  in_int_t v[N];
  out_int_t out[N];

  for (int y = 0; y < N; ++y) {
    v[y] = rand() % 100;
    for (int x = 0; x < N; ++x) {
      m[y][x] = rand() % 100;
    }
  }

  CALL_KERNEL(matvec, m, v, out);
  return 0;
}
