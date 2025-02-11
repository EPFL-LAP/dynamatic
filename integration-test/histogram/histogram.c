//===- histogram.c ---------------------------------------------*- C -*-===//

#include "dynamatic/Integration.h"
#include "histogram.h"
#include <stdlib.h>

void histogram(in_int_t feature[10], in_float_t weight[10],
               inout_float_t hist[10], in_int_t n) {
  for (int i = 0; i < n; ++i) {
    int m = feature[i];
    float wt = weight[i];
    float x = hist[m];
    hist[m] = x + wt;
  }
}

int main(void) {
  in_int_t feature[10];
  in_float_t weight[10];
  inout_float_t hist[10];
  in_int_t n;

  n = 10;
  // for (int i = 0; i < 10; ++i) {
  //   feature[i] = rand() % 10;
  //   weight[i] = rand() % 100;
  //   hist[i] = rand() % 100;
  // }
  for (int i = 0; i < 10; ++i) {
    feature[i] = (i+3) % 10;
    weight[i] = i * 20;
    hist[i] = i * 10;
  }

  CALL_KERNEL(histogram, feature, weight, hist, n);
  return 0;
}
