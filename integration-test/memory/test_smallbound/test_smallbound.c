#include "dynamatic/Integration.h"
#include "test_smallbound.h"
#include <stdlib.h>

void test_smallbound(inout_int_t x[N][M], inout_int_t y[N]) {
  for (unsigned i = 0; i < N; ++i)
    for (unsigned j = 0; j < M; ++j)
      x[i][j] = y[i] + x[0][0];
}

int main(void) {
  inout_int_t x[N][M];
  in_int_t y[N];

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      x[i][j] = rand() % 100;
    }
  }

  for (int j = 0; j < N; ++j) {
    y[j] = rand() % 100;
  }

  CALL_KERNEL(test_smallbound, x, y);
  return 0;
}
