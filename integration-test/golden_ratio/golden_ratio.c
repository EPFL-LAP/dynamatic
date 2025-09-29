#include "dynamatic/Integration.h"
#include "stdbool.h"
#include <math.h>

#define ITERS 100
float golden_ratio(float x0, float x1) {
  float x = x0;
  float original_x = x1;
  for (int i = 0; i < ITERS; i++) {
    float old_x;
    do {
      old_x = x;
      x = 0.5f * (x + x * original_x);
    } while (fabs(x - old_x) >= 0.1f);
    x += 1.0f;
    original_x = 1 / x;
  }
  return x;
}

int main(void) {
  float x0 = 100.0f;
  float x1 = 0.01f;
  CALL_KERNEL(golden_ratio, x0, x1);
  return 0;
}
