#include "dynamatic/Integration.h"
#include "stdlib.h"
#include <math.h>

#define LOG_ITERS 10
float newton(float y) {
  float c = 1000.0f;
  float x1 = 0.0f;
  float x0 = y;
  float a = 1e-30f;
  while (c >= a) {
    x1 = 0.5f * (x0 + 2.0f / x0);
    c = fabs(x0 - x1);
    x0 = x1;
  }
  return x1;
}

int main(void) {
  float y = 1e20f;
  CALL_KERNEL(newton, y);
  return 0;
}
