#include "test_memory_7.h"
#include "../integration_utils.h"
#include <stdlib.h>

void test_memory_7(inout_int_t a[N], in_int_t n) {
  for (int i = 2; i < n; i++) {
    a[i] = a[i - 1] + a[i - 2] + 5;
    if (a[i] > 0)
      a[i] = 0;
  }
}

int main(void) {
  inout_int_t a[N];
  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 100;

  CALL_KERNEL(test_memory_7, a, N);
  return 0;
}
