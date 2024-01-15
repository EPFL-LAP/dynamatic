#include "test_memory_11.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void test_memory_11(int a[N]) {
  for (unsigned i = 1; i < N; i++)
    a[i] = a[i - 1] + a[i];
  for (unsigned i = 1; i < N; i++)
    a[i] = a[i - 1];
}

int main(void) {
  inout_int_t a[N];

  srand(13);
  for (unsigned j = 0; j < N; ++j)
    a[j] = (rand() % 100) - 50;

  CALL_KERNEL(test_memory_11, a);
  return 0;
}
