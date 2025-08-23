#include "dynamatic/Integration.h"
#include <stdlib.h>

// clang-format off

// No LSQ needed:
// - read(a[1]) --(data)--> write(a[0]) --(same port)--> write(a[1])
// Therefore the order "read(a[i]) --> write(a[i])" is always honored

// clang-format on

#define N 5
void test_memory_2(int a[N], int n) {
  for (int i = 0; i < n - 1; i++)
    a[i] = a[i + 1] + N;
}

int main(void) {
  int a[N];
  int n = N;

  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(test_memory_2, a, n);
  return 0;
}
