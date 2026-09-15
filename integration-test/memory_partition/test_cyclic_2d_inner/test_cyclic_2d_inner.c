#include <stdlib.h>
#define ROWS 4
#define COLS 8
#include "dynamatic/Integration.h"

void test_cyclic_2d_inner(const int A[ROWS][COLS], const int B[ROWS][COLS],
                          const int C[ROWS][COLS], int result[ROWS][COLS]) {
#pragma DYN array_partition array = intermediate dimension = 2 style =         \
    cyclic factor = 2
  int intermediate[ROWS][COLS];
#pragma clang loop unroll_count(2)
  for (int row = 0; row < ROWS; row++) {
    for (int col = 0; col < COLS; col++) {
      intermediate[row][col] = A[row][col] * B[row][col];
    }
  }
  for (int row = 0; row < ROWS; row++) {
    for (int col = 0; col < COLS; col++) {
      result[row][col] = intermediate[row][col] * C[row][col];
    }
  }
}

int main(void) {
  int A[ROWS][COLS];
  int B[ROWS][COLS];
  int C[ROWS][COLS];
  int result[ROWS][COLS];
  for (int row = 0; row < ROWS; row++) {
    for (int col = 0; col < COLS; col++) {
      A[row][col] = rand() % 100;
      B[row][col] = rand() % 100;
      C[row][col] = rand() % 100;
    }
  }
  CALL_KERNEL(test_cyclic_2d_inner, A, B, C, result);
  return 0;
}
