#include "test_memory_10.h"
#include "../integration_utils.h"
#include <stdlib.h>

void test_memory_10(inout_int_t a[N], in_int_t n) {
  for (int i = 0; i < n - 1; i++)
    a[i] = a[i] + a[i + 1] + 5;
}

int main(void) {
  inout_int_t a[N];
  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 100;

  CALL_KERNEL(test_memory_10, a, N);
  return 0;
}
