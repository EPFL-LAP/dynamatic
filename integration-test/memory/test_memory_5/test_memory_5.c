#include "dynamatic/Integration.h"
#include <stdlib.h>
#define N 4

// Needs an LSQ:
// WAR: "write(a[i]) @ iter. i" --> "read(a[i]) @ iter. i + 1"

void test_memory_5(int a[N], int n) {
  for (int i = 2; i < n; i++)
    a[i] = a[i - 1] + a[i - 2] + 5;
}

int main(void) {
  int a[N];
  int n = N;

  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(test_memory_5, a, n);
  return 0;
}
