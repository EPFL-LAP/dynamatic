
#include "test_memory_1.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void test_memory_1(inout_int_t a[N], in_int_t n) {
  for (int i = 0; i < n; i++)
    a[i] = a[i] + 5;
}

int main(void) {
  inout_int_t a[N];
  in_int_t n = N;

  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(test_memory_1, a, n);
  return 0;
}
