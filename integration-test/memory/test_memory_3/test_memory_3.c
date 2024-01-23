#include "test_memory_3.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void test_memory_3(inout_int_t a[N], in_int_t n) {
  for (int i = 0; i < n; i++) {
    if (a[i] == n)
      a[i] = 0;
  }
}

int main(void) {
  inout_int_t a[N];
  in_int_t n = N;

  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand();

  CALL_KERNEL(test_memory_3, a, n);
  return 0;
}
