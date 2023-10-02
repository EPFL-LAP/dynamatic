//===- fir.c - Computes FIR of two integer arrays -----------------*- C -*-===//
//
// Declares the fir kernel which computes a finite impulse response (FIR)
// between two discrete signals.
//
//===----------------------------------------------------------------------===//

#include "fir.h"

int fir(int di[N], int idx[N]) {
  int tmp = 0;
  for (unsigned i = 0; i < N; i++)
    tmp += idx[i] * di[N_DEC - i];
  return tmp;
}
