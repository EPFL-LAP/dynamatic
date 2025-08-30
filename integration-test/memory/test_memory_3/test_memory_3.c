#include "dynamatic/Integration.h"
#include <stdlib.h>
#define N 4

// NOTE: this technically doesn't need an LSQ
// But the loop is not SCOP, so index analysis doesn't work
// The memory analysis identifies a spurious RAW:
// write(a[i]) --> read(a[i + 1])

void test_memory_3(int a[N], int n) {
  for (int i = 0; i < n; i++) {
    if (a[i] == n)
      a[i] = 0;
  }
}

int main(void) {
  int a[N];
  int n = N;

  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand();

  CALL_KERNEL(test_memory_3, a, n);
  return 0;
}
