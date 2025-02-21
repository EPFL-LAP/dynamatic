//===- histogram.c ---------------------------------------------*- C -*-===//

#include "dynamatic/Integration.h"
#include "histogram_2.h"
#include <stdlib.h>

void histogram_2(in_int_t feature[1000], in_float_t weight[1000],
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
    feature[i] = (i+3) % 1000;
    weight[i] = i +1;
    hist[i] = i * 10;
  }

  CALL_KERNEL(histogram_2, feature, weight, hist, n);
  return 0;
}
