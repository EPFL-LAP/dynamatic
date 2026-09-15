#include <stdlib.h>
#define N 16
#include "dynamatic/Integration.h"

// Complete partition: factor is derived internally as totalSize, so every
// element gets its own single-element bank.
void test_complete_1d(const int A[N], const int B[N], const int C[N],
                      int result[N]) {
#pragma DYN array_partition array = intermediate dimension = 1 style =         \
    complete factor = 1
  int intermediate[N];
#pragma clang loop unroll_count(16)
  for (int i = 0; i < N; ++i) {
    intermediate[i] = A[i] * B[i];
  }
#pragma clang loop unroll_count(16)
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
  CALL_KERNEL(test_complete_1d, A, B, C, result);
  return 0;
}
