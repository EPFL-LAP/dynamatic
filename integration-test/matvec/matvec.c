//===- matvec.c - Multiply matrix to vector  ----------------------*- C -*-===//
//
// Implements the matvec kernel.
//
//===----------------------------------------------------------------------===//

#include "matvec.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int matvec(in_int_t v[N]) {
  int tmp = 1;
  for (unsigned i = 0; i < N; i++) {
    tmp = v[i];
    for (unsigned j = 0; j < N; j++) {
      tmp *= v[j];
    }
  }
  return tmp;

  // int tmp2 = 1;
  // for (unsigned i = 0; i < N; i++) {
  //   //tmp = 0;
  //   //for (unsigned j = 0; j < N; j++) {
  //   tmp2 *= v[i] + m[i][i];
  //   //}
  //   //out[i] = tmp;
  // }
  // return tmp + tmp2;
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

  CALL_KERNEL(matvec, v);
  return 0;
}
