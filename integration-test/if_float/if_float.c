#include "dynamatic/Integration.h"
#include "stdbool.h"
#include <math.h>

#define ITERS 100
float if_float(float x0, float minus_trace[ITERS]) {
  float x = x0;
  for (int i = 0; i < ITERS; i++) {
    if (x * x - x <= 0.0f) {
      x += 1;
    } else {
      minus_trace[i] = x;
      x -= 1;
    }
    x = 1 / x;
  }
  return x;
}

int main(void) {
  float x0 = 100.0f;
  float minus_trace[ITERS];
  CALL_KERNEL(if_float, x0, minus_trace);
  return 0;
}
