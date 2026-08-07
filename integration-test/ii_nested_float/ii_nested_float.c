//===- ii_nested_float.c - Nested loop with a slow recurrence -----*- C -*-===//
//
// Kernel whose inner loop carries a floating-point accumulation: a recurrence
// of several cycles, run for only a few iterations per activation of the
// frequently re-entered inner loop.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 20
#define M 5

/// Sums each row of a matrix into a vector of per-row sums.
void ii_nested_float(float a[N][M], float sums[N]) {
  for (unsigned i = 0; i < N; ++i) {
    float acc = 0.0f;
    for (unsigned j = 0; j < M; ++j)
      acc += a[i][j];
    sums[i] = acc;
  }
}

int main(void) {
  float a[N][M];
  float sums[N];

  for (int y = 0; y < N; ++y)
    for (int x = 0; x < M; ++x)
      a[y][x] = (float)(rand() % 1000) / 8.0f;

  CALL_KERNEL(ii_nested_float, a, sums);
  return 0;
}
