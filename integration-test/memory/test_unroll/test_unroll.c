#define N 20
#include "dynamatic/Integration.h"

#define UNROLL_FACTOR 4

void test_unroll(const int A[N], const int B[N], const int C[N],
                 int result[N]) {
  // NOTE: The "array-partition" pass replicate this array to allow
  // `UNROLL_FACTOR` concurrent accesses.
  int intermediate[N];
#pragma clang loop unroll_count(UNROLL_FACTOR)
  for (int i = 0; i < N; i++) {
    intermediate[i] = A[i] * B[i];
  }
#pragma clang loop unroll_count(UNROLL_FACTOR)
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
    A[i] = i;
    B[i] = i;
    C[i] = i;
  }

  CALL_KERNEL(test_unroll, A, B, C, result);
  return 0;
}
