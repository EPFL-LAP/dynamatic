//===- bicg.c - BiCGSTAB (BiConjugate Gradient STABilized method) -*- C -*-===//
//
// Implements the bicg kernel.
//
// This file is partially taken from the Polybench 4.2.1 testsuite.
// https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/tree/master
//
//===----------------------------------------------------------------------===//

#include "bicg.h"
#include <cstddef>

int bicg(int a[N][N], int s[N], int q[N], int p[N], int r[N]) {
  int tmp = 0;
  for (size_t i = 0; i < N; i++) {
    tmp = q[i];

    for (size_t j = 0; j < N; j++) {
      int val = a[i][j];
      s[j] = s[j] + r[i] * val;
      tmp += val * p[j];
    }

    q[i] = tmp;
  }
  return tmp;
}
