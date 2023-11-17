//===- bicg.c - BiCGSTAB (BiConjugate Gradient STABilized method) -*- C -*-===//
//
// Implements the bicg kernel.
//
// This file is partially taken from the Polybench 4.2.1 testsuite.
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/tree/master
//
//===----------------------------------------------------------------------===//

#include "bicg.h"
#include "../integration_utils.h"
#include "stdlib.h"

int bicg(in_int_t a[N][N], inout_int_t s[N], inout_int_t q[N], in_int_t p[N],
         in_int_t r[N]) {
  int tmp = 0;
  for (unsigned i = 0; i < N; i++) {
    tmp = q[i];

    for (unsigned j = 0; j < N; j++) {
      int val = a[i][j];
      s[j] = s[j] + r[i] * val;
      tmp += val * p[j];
    }

    q[i] = tmp;
  }
  return tmp;
}

int main(void) {
  in_int_t a[N][N];
  inout_int_t s[N];
  inout_int_t q[N];
  in_int_t p[N];
  in_int_t r[N];

  for (int y = 0; y < N; ++y) {
    s[y] = rand() % 100;
    q[y] = rand() % 100;
    p[y] = rand() % 100;
    r[y] = rand() % 100;
    for (int x = 0; x < N; ++x) {
      a[y][x] = rand() % 100;
    }
  }

  CALL_KERNEL(bicg, a, s, q, p, r);
  return 0;
}
