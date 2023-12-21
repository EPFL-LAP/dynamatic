
#include "test_memory_4.h"
#include "../integration_utils.h"
#include <stdlib.h>

void test_memory_4(inout_int_t a[N], in_int_t n) {
  int x = 0;
  for (int i = 0; i < n - 1; i++) {
    a[i] = x;
    x = a[i + 1];
  }
}

int main(void) {
  inout_int_t a[N];
  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(test_memory_4, a, N);
  return 0;
}
