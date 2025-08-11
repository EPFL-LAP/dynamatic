#define N 4
#include "dynamatic/Integration.h"
#include <stdlib.h>

// NOTE: No LSQ needed:
// WAR: read(a[i]) --(data dep)--> write(a[i])
// Always enforced by data dependency.

void test_memory_1(int a[N], int n) {
  for (int i = 0; i < n; i++)
    a[i] = a[i] + 5;
}

int main(void) {
  int a[N];
  int n = N;

  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(test_memory_1, a, n);
  return 0;
}
