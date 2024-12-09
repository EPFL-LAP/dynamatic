//===- min_lsq_example.c ---------------------------------------------*- C -*-===//

#include "dynamatic/Integration.h"
#include "min_lsq_example.h"
#include <stdlib.h>

void min_lsq_example(in_int_t x[100], in_int_t y[100], in_int_t n) {
  for (int i = 1; i < n; ++i)
    x[i] = x[0] + x[i] * y[i];
}

int main(void) {
  in_int_t x[100], y[100];
  in_int_t n;

  n = 100;
  for (int i = 0; i < 100; ++i) {
    x[i] = rand() % 100;
    y[i] = rand() % 100;
  }

  CALL_KERNEL(min_lsq_example, x, y, n);
  return 0;
}
