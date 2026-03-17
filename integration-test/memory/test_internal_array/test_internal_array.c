#include <stdlib.h>
#define N 20
#define M 32

#include "dynamatic/Integration.h"

#define UNROLL_FACTOR 4

void test_internal_array(const int A[N][M], const int B[N][M],
                         const int C[N][M], int result[N][M]) {
  // NOTE: The "array-partition" pass replicate this array to allow 2 concurrent
  // accesses
  int intermediate[N][M];
  for (int i = 0; i < N; ++i) {
    // #pragma clang loop unroll_count(UNROLL_FACTOR)
    for (int j = 0; j < M; ++j)
      intermediate[i][j] = A[i][j] * B[i][j];
  }
  for (int i = 0; i < N; i++) {
    // #pragma clang loop unroll_count(UNROLL_FACTOR)
    for (int j = 0; j < M; ++j)
      result[i][j] = intermediate[i][j] * C[i][j];
  }
}

int main(void) {
  int A[N][M];
  int B[N][M];
  int C[N][M];
  int result[N][M];

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      A[i][j] = rand() % 100;
      B[i][j] = rand() % 100;
      C[i][j] = rand() % 100;
    }
  }

  CALL_KERNEL(test_internal_array, A, B, C, result);
  return 0;
}
