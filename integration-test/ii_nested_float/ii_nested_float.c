//===- ii_nested_float.c - Nested loop with a slow recurrence -----*- C -*-===//
//
// Implements the ii_nested_float kernel.
//
//===----------------------------------------------------------------------===//

#include "ii_nested_float.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

void ii_nested_float(in_float_t a[N][M], out_float_t sums[N]) {
  for (unsigned i = 0; i < N; ++i) {
    float acc = 0.0f;
    for (unsigned j = 0; j < M; ++j)
      acc += a[i][j];
    sums[i] = acc;
  }
}

int main(void) {
  in_float_t a[N][M];
  out_float_t sums[N];

  for (int y = 0; y < N; ++y)
    for (int x = 0; x < M; ++x)
      a[y][x] = (float)(rand() % 1000) / 8.0f;

  CALL_KERNEL(ii_nested_float, a, sums);
  return 0;
}
