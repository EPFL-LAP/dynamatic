#include <stdlib.h>
#define N 67
#include "dynamatic/Integration.h"

// NOTE: Test for division with remainder after doing N / factor
void test_block_1d_factor8(const int A[N], const int B[N], const int C[N],
                          int result[N]) {
#pragma DYN array_partition array = intermediate dimension = 1 style =         \
    block factor = 8
  int intermediate[N];
  for (int i = 0; i < N; ++i) {
    intermediate[i] = A[i] * B[i];
  }
  for (int i = 0; i < N; i++) {
    result[i] = intermediate[i] * C[i];
  }
}

int main(void) {
  int A[N];
  int B[N];
  int C[N];
  int result[N];
  for (int i = 0; i < N; ++i) {
    A[i] = rand() % 100;
    B[i] = rand() % 100;
    C[i] = rand() % 100;
  }
  CALL_KERNEL(test_block_1d_factor8, A, B, C, result);
  return 0;
}
