#include "test_memory_12.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void test_memory_12(int a[N]) {
  for (unsigned i = 0; i < N; i++) {
    a[0] = 0;
    a[1] = 1;
    a[2] = 2;
    if (i & 1)
      a[3] = 3;
  }
}

int main(void) {
  inout_int_t a[N];

  srand(13);
  for (unsigned i = 0; i < N; ++i) {
    a[i] = (rand() % 100) - 50;
  }

  CALL_KERNEL(test_memory_12, a);
  return 0;
}
