#include "memory_loop.h"
#include "../integration_utils.h"
#include <stdlib.h>

void memory_loop(inout_int_t x[N], in_int_t y[N]) {
  for (unsigned i = 1; i < N; ++i)
    x[i] = x[i] * y[i] + x[0];
}

int main(void) {
  inout_int_t x[N];
  in_int_t y[N];

  for (int j = 0; j < N; ++j) {
    x[j] = rand() % 100;
    y[j] = rand() % 100;
  }

  CALL_KERNEL(memory_loop, x, y);
  return 0;
}
