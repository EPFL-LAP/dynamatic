//===- stencil_2d.c - Grid-based computation ----------------------*- C -*-===//
//
// Implements the stencil_2d kernel.
//
//===----------------------------------------------------------------------===//

#include "stencil_2d.h"
#include "../integration_utils.h"
#include "stdlib.h"

int stencil_2d(in_int_t orig[N], in_int_t filter[M], out_int_t sol[N]) {
  int temp = 0;
  for (unsigned c = 0; c < 28; c++) {
    temp = 0;
    for (unsigned k1 = 0; k1 < 3; k1++)
      for (unsigned k2 = 0; k2 < 3; k2++)
        temp += filter[k1 * 3 + k2] * orig[k1 * 30 + c + k2];
    sol[c] = temp;
  }
  return temp;
}

int main(void) {
  in_int_t orig[N];
  in_int_t filter[10];
  out_int_t sol[N];

  for (int j = 0; j < N; ++j)
    orig[j] = rand() % 100;
  for (int j = 0; j < M; ++j)
    filter[j] = rand() % 100;

  CALL_KERNEL(stencil_2d, orig, filter, sol);
  return 0;
}
