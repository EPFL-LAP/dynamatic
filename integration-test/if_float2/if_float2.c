#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"
#include <math.h>

#define ITERS 100
float if_float2(float x0, float a[ITERS], float minus_trace[ITERS]) {
  float x = x0;
  for (int i = 0; i < ITERS; i++) {
    if (a[i] * x + (-0.6f) * x <= 0.0f) {
      x *= 1.1f;
    } else {
      minus_trace[i] = x;
      x *= 1.2f;
    }
    x = 1 / x;
  }
  return x;
}

int main(void) {
  float x0 = 100.0f;
  float a[ITERS];
  float minus_trace[ITERS];

  srand(13);
  for (int j = 0; j < ITERS; ++j) {
    a[j] = (float)(rand() % 100) / 100.0f;
  }

  CALL_KERNEL(if_float2, x0, a, minus_trace);
  return 0;
}
