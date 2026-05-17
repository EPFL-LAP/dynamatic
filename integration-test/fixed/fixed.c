// clang-format off
#include "fixed.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

float fixed(in_float_t y) {
  float c = 1000.0f;
  float x1 = 0.0f;
  float x0 = 1.0f;
  float a = 0.00000001f;
  bool loopAgain;
  do {
    x1 = x0 * y;
    c = x0 - x1;
    x0 = x1;
    loopAgain = c >= a;
    #pragma DYN speculate variable=loopAgain max_predictions=3 style=standard
  } while (loopAgain);
  return x1;
}

int main(void) {
  in_float_t y = 0.7f;
  CALL_KERNEL(fixed, y);
  return 0;
}
