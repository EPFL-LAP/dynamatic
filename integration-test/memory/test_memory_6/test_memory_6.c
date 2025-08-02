#define N 4
#include "dynamatic/Integration.h"
#include <stdlib.h>

// Needs an LSQ:
// "a[i - 1] = 0" has RAW on both reads in "a[i] = a[i - 1] + a[i - 2] + 5"
// "a[i] = ..." has RAW on both reads in the same statement.

void test_memory_6(int a[N], int n) {
  for (int i = 2; i < n; i++) {
    a[i] = a[i - 1] + a[i - 2] + 5;
    a[i - 1] = 0;
  }
}

int main(void) {
  int a[N];
  int n = N;

  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(test_memory_6, a, n);
  return 0;
}
