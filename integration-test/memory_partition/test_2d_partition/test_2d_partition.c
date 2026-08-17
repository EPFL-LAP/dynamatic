#include <stdlib.h>
#define ROWS 32
#define COLS 64
#include "dynamatic/Integration.h"

// NOTE: Test for outer partition of 2d array, i.e. A[N][M] -> A1[N/2][M], A2[N/2][M]
void test_2d_partition(const int A[ROWS][COLS], const int B[ROWS][COLS],
                       const int C[ROWS][COLS], int result[ROWS][COLS]) {
#pragma DYN array_partition array = intermediate dimension = 1 style =         \
    block factor = 2
  int intermediate[ROWS][COLS];
  for (int i = 0; i < ROWS; ++i) {
    for (int j = 0; j < COLS; ++j) {
      intermediate[i][j] = A[i][j] * B[i][j];
    }
  }
  for (int i = 0; i < ROWS; ++i) {
    for (int j = 0; j < COLS; ++j) {
      result[i][j] = intermediate[i][j] * C[i][j];
    }
  }
}

int main(void) {
  int A[ROWS][COLS];
  int B[ROWS][COLS];
  int C[ROWS][COLS];
  int result[ROWS][COLS];
  for (int i = 0; i < ROWS; ++i) {
    for (int j = 0; j < COLS; ++j) {
      A[i][j] = rand() % 100;
      B[i][j] = rand() % 100;
      C[i][j] = rand() % 100;
    }
  }
  CALL_KERNEL(test_2d_partition, A, B, C, result);
  return 0;
}
