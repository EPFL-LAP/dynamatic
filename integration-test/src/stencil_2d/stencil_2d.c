//===- stencil_2d.c - Grid-based computation ----------------------*- C -*-===//
//
// Implements the stencil_2d kernel.
//
//===----------------------------------------------------------------------===//

#include "stencil_2d.h"
#include <cstddef>

int stencil_2d(int orig[N], int filter[M], int sol[N]) {
  int temp = 0;
  for (size_t c = 0; c < 28; c++) {
    temp = 0;
    for (size_t k1 = 0; k1 < 3; k1++)
      for (size_t k2 = 0; k2 < 3; k2++)
        temp += filter[k1 * 3 + k2] * orig[k1 * 30 + c + k2];
    sol[c] = temp;
  }
  return temp;
}
