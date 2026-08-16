//===- ii_nested_int.c - Nested loop with an II=1 recurrence ------*- C -*-===//
//
// Kernel whose inner loop carries an integer accumulation: a combinational
// recurrence that pipelines at II=1.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 8
#define M 30

/// Sums each row of a matrix into a vector of per-row sums.
void ii_nested_int(int a[N][M], int sums[N]) {
  for (unsigned i = 0; i < N; ++i) {
    int acc = 0;
    for (unsigned j = 0; j < M; ++j)
      acc += a[i][j];
    sums[i] = acc;
  }
}

int main(void) {
  int a[N][M];
  int sums[N];

  for (int y = 0; y < N; ++y)
    for (int x = 0; x < M; ++x)
      a[y][x] = rand() % 100;

  CALL_KERNEL(ii_nested_int, a, sums);
  return 0;
}
