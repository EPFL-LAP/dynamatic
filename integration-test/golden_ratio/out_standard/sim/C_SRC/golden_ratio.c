#include "dynamatic/Integration.h"
#include "stdbool.h"
#include <math.h>

#define ITERS 100
float golden_ratio(float x0) {
  float x = x0;
  float original_x = 1 / x;
  for (int i = 0; i < ITERS; i++) {
    while (true) {
      float next_x = 0.5f * (x + x * original_x);
      if (fabs(next_x - x) < 0.1f)
        break;
      x = next_x;
    }
    x += 1.0f;
    original_x = 1 / x;
  }
  return x;
}

int main(void) {
  float x0 = 100.0f;
  CALL_KERNEL(golden_ratio, x0);
  return 0;
}
