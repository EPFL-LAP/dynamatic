#include "dynamatic/Integration.h"
#include "stdlib.h"

#define LOG_ITERS 10
float fixed_log(float y, float dump[LOG_ITERS]) {
  float c = 1000.0f;
  float x1 = 0.0f;
  float x0 = 1.0f;
  float a = 0.00000001f;
  int iter = 0;
  while (c >= a) {
    x1 = x0 * y;
    c = x0 - x1;
    x0 = x1;
    if (iter < LOG_ITERS) {
      dump[iter] = x1;
      iter++;
    }
  }
  return x1;
}

int main(void) {
  float y = 0.7f;
  float dump[LOG_ITERS];
  CALL_KERNEL(fixed_log, y, dump);
  return 0;
}
