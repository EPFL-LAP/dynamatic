#include <stdlib.h>
#define N 8
#define M 4
#include "dynamatic/Integration.h"

const int A[N][M] = {{1, 2, 3, 4},     {5, 6, 7, 8},     {9, 10, 11, 12},
                     {13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24},
                     {25, 26, 27, 28}, {29, 30, 31, 32}};

void test_cyclic_2d_global(const int B[N][M], const int C[N][M],
                           int result[N][M]) {
  int intermediate[N][M];
#pragma DYN array_partition array = A dimension = 2 style = cyclic factor = 2
#pragma clang loop unroll_count(2)
  for (int i = 0; i < N; ++i) {
#pragma clang loop unroll_count(2)
    for (int j = 0; j < M; ++j) {
      intermediate[i][j] = A[i][j] * B[i][j];
    }
  }

#pragma clang loop unroll_count(2)
  for (int i = 0; i < N; ++i) {
#pragma clang loop unroll_count(2)
    for (int j = 0; j < M; ++j) {
      result[i][j] = intermediate[i][j] * C[i][j];
    }
  }
}

int main(void) {
  int B[N][M];
  int C[N][M];
  int result[N][M];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      B[i][j] = rand() % 100;
      C[i][j] = rand() % 100;
    }
  }
  CALL_KERNEL(test_cyclic_2d_global, B, C, result);
  return 0;
}
