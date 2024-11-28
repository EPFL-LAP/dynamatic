//===- fir.c - Computes FIR of two integer arrays -----------------*- C -*-===//
//
// Declares the fir kernel which computes a finite impulse response (FIR)
// between two discrete signals.
//
//===----------------------------------------------------------------------===//

#include "float_for.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int float_for(in_int_t di[N], in_int_t idx[N]) {
  int tmp = 0;
  int i = 0;
  for (float j = 0.0f; j < 4.5f; j += 1.0f) {
    tmp += idx[i] * di[N_DEC - i];
    i += 1;
  }
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

  CALL_KERNEL(float_for, di, idx);
  return 0;
}
