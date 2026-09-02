#include <stdlib.h>
#define N 32
#include "dynamatic/Integration.h"

void test_cyclic_1d_factor3(const int A[N], const int B[N], const int C[N],
                            int result[N]) {
#pragma DYN array_partition array = intermediate dimension = 1 style =         \
    cyclic factor = 3
  int intermediate[N];
#pragma clang loop unroll_count(3)
  for (int i = 0; i < N; ++i) {
    intermediate[i] = A[i] * B[i];
  }
#pragma clang loop unroll_count(3)
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
  CALL_KERNEL(test_cyclic_1d_factor3, A, B, C, result);
  return 0;
}
