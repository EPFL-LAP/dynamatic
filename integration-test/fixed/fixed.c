#include "dynamatic/Integration.h"
#include "stdlib.h"

float fixed(float y) {
  float c = 1000.0f;
  float x1 = 0.0f;
  float x0 = 1.0f;
  float a = 0.00000001f;
  while (c >= a) {
    x1 = x0 * y;
    c = x0 - x1;
    x0 = x1;
  }
  return x1;
}

int main(void) {
  float y = 0.7f;
  CALL_KERNEL(fixed, y);
  return 0;
}
