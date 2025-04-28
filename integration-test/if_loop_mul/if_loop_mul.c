#include "if_loop_mul.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

float if_loop_mul(in_float_t a[1000], in_float_t b[1000]) {
  int i;
  float dist;
  float sum = 1.0;

  for (i = 0; i < 1000; i++) {
    dist = a[i] - b[i];

    if (dist >= 0) {

      sum = (sum * dist);
    }
  }
  return sum;
}

int main(void) {
  in_float_t a[1000];
  in_float_t b[1000];

  for (int i = 0; i < 1000; ++i) {
    a[i] = (float)i;
    b[i] = (float)i + 10;

    if (i % 100 == 0)
      b[i] = 0;
  }

  CALL_KERNEL(if_loop_mul, a, b);
  return 0;
}
