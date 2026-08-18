#include <stdlib.h>
#define N 32
#include "dynamatic/Integration.h"

// NOTE: Tests for two bugs:
// 1. Reuse of GEP by two different indices
// 2. Zero indexing not producing a GEP since we simply load the address
void test_zero_elem(const int A[N], int result[1], int idx) {
#pragma DYN array_partition array = intermediate dimension = 1 style =         \
    block factor = 2
  int intermediate[N];
  intermediate[idx] = A[idx];
  intermediate[0] = A[0];
  *result = intermediate[0] + intermediate[idx];
}

int main(void) {
  int A[N];
  int result[1];
  for (int i = 0; i < N; ++i) {
    A[i] = rand() % 100;
  }
  CALL_KERNEL(test_zero_elem, A, result, rand() % N);
  return 0;
}
