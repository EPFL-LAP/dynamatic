#include "test_memory_5.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void test_memory_5(inout_int_t a[N], in_int_t n) {
  for (int i = 2; i < n; i++)
    a[i] = a[i - 1] + a[i - 2] + 5;
}

int main(void) {
  inout_int_t a[N];
  in_int_t n = N;

  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(test_memory_5, a, n);
  return 0;
}
