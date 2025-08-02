#define N 4
#include "dynamatic/Integration.h"
#include <stdlib.h>

// Needs an LSQ:
// RAW: "write(a[i+1]) @ iter. i" --> "read(a[i+1]) @ iter. i + 1"

void test_memory_4(int a[N], int n) {
  int x = 0;
  for (int i = 0; i < n - 1; i++) {
    a[i] = x;
    x = a[i + 1];
  }
}

int main(void) {
  int a[N];
  int n = N;

  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(test_memory_4, a, n);
  return 0;
}
