#include <stdlib.h>
#define N 32
#include "dynamatic/Integration.h"
const int A[N] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

void test_cyclic_1d_global(const int B[N], const int C[N], int result[N]) {
  int intermediate[N];

#pragma DYN array_partition array = A dimension = 1 style = cyclic factor = 2

#pragma clang loop unroll_count(2)
  for (int i = 0; i < N; ++i) {
    intermediate[i] = A[i] * B[i];
  }
#pragma clang loop unroll_count(2)
  for (int i = 0; i < N; i++) {
    result[i] = intermediate[i] * C[i];
  }
}

int main(void) {
  int B[N];
  int C[N];
  int result[N];
  for (int i = 0; i < N; ++i) {
    B[i] = rand() % 100;
    C[i] = rand() % 100;
  }
  CALL_KERNEL(test_cyclic_1d_global, B, C, result);
  return 0;
}
