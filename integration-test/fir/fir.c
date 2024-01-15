//===- fir.c - Computes FIR of two integer arrays -----------------*- C -*-===//
//
// Declares the fir kernel which computes a finite impulse response (FIR)
// between two discrete signals.
//
//===----------------------------------------------------------------------===//

#include "fir.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int fir(in_int_t di[N], in_int_t idx[N]) {
  int tmp = 0;
  for (unsigned i = 0; i < N; i++)
    tmp += idx[i] * di[N_DEC - i];
  return tmp;
}

int main(void) {
  in_int_t di[N];
  in_int_t idx[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    di[j] = rand() % 100;
    idx[j] = rand() % 100;
  }

  CALL_KERNEL(fir, di, idx);
  return 0;
}
