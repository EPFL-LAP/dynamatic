#include <stdlib.h>
#define N 32
#include "dynamatic/Integration.h"

void test_cyclic_1d_two_arrays_factor(const int A[N], const int B[N],
                                      const int C[N], int result[N]) {
#pragma DYN array_partition array = prodAB dimension = 1 style =               \
    cyclic factor = 2
  int prodAB[N];
#pragma DYN array_partition array = prodBC dimension = 1 style =               \
    cyclic factor = 4
  int prodBC[N];

#pragma clang loop unroll_count(4)
  for (int i = 0; i < N; ++i) {
    prodAB[i] = A[i] * B[i];
    prodBC[i] = B[i] * C[i];
  }
#pragma clang loop unroll_count(4)
  for (int i = 0; i < N; i++) {
    result[i] = prodAB[i] + prodBC[i];
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
  CALL_KERNEL(test_cyclic_1d_two_arrays_factor, A, B, C, result);
  return 0;
}
