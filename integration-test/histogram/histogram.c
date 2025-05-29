//===- histogram.c ---------------------------------------------*- C -*-===//

#include "histogram.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void histogram(in_int_t feature[1000], in_float_t weight[1000],
               inout_float_t hist[1000], in_int_t n) {
  for (int i = 0; i < n; ++i) {
    int m = feature[i];
    float wt = weight[i];
    float x = hist[m];
    hist[m] = x + wt;
  }
}

int main(void) {
  in_int_t feature[1000];
  in_float_t weight[1000];
  inout_float_t hist[1000];
  in_int_t n;

  n = 1000;
  for (int i = 0; i < 1000; ++i) {
    feature[i] = rand() % 1000;
    weight[i] = rand() % 100;
    hist[i] = rand() % 100;
  }

  CALL_KERNEL(histogram, feature, weight, hist, n);
  return 0;
}
