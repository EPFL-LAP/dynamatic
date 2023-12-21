#include "test_memory_9.h"
#include "../integration_utils.h"
#include <stdlib.h>

void test_memory_9(inout_int_t a[N], in_int_t n) {
  for (int i = 1; i < n; i++)
    a[i] = 5 * a[i - 1];
}

int main(void) {
  inout_int_t a[N];
  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 100;

  CALL_KERNEL(test_memory_9, a, N);
  return 0;
}
