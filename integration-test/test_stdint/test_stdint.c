#include "dynamatic/Integration.h"
#include "test_stdint.h"
#include <stdlib.h>

void test_stdint(inout_int8_t a[N], in_int8_t b) {
  for (unsigned i = 0; i < N; ++i) {
    in_int8_t x = a[i];
    a[i] = x * x * x + b;
  }
}

int main(void) {
  in_int8_t a[N];
  for (unsigned j = 0; j < N; ++j)
    a[j] = j;
  in_int8_t b = rand() % 64;

  CALL_KERNEL(test_stdint, a, b);
  return 0;
}
