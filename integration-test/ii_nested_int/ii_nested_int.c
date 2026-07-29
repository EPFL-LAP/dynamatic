//===- ii_nested_int.c - Nested loop with an II=1 recurrence ------*- C -*-===//
//
// Implements the ii_nested_int kernel.
//
//===----------------------------------------------------------------------===//

#include "ii_nested_int.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

void ii_nested_int(in_int_t a[N][M], out_int_t sums[N]) {
  for (unsigned i = 0; i < N; ++i) {
    int acc = 0;
    for (unsigned j = 0; j < M; ++j)
      acc += a[i][j];
    sums[i] = acc;
  }
}

int main(void) {
  in_int_t a[N][M];
  out_int_t sums[N];

  for (int y = 0; y < N; ++y)
    for (int x = 0; x < M; ++x)
      a[y][x] = rand() % 100;

  CALL_KERNEL(ii_nested_int, a, sums);
  return 0;
}
