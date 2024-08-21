//===- gsumif.c - Testing long-latency--loop-carried dependency ---*- C -*-===//
//
// Implements a kernel with long-latency--loop-carried dependency in the loop
// body
//
// Author: Jianyi Cheng, DSS
// https://zenodo.org/record/3561115
//
//===---------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "gsumif.h"
#include <stdlib.h>

float gsumif(in_float_t a[1000]) {
  int i;
  float d;
  float s = 0.0;

  for (i = 0; i < 1000; i++) {
    d = a[i];
    if (d >= 0) {
      float p;
      if (i > 5)
        p = ((d + (float)0.25) * d + (float)0.5) * d + (float)0.125;
      else
        p = ((d + (float)0.64) * d + (float)0.7) * d + (float)0.21;
      s += p;
    }
  }
  return s;
}

#define AMOUNT_OF_TEST 1

int main(void) {
  in_float_t a[1000];
  in_float_t b[1000];

  for (int i = 0; i < 1000; ++i) {
    a[i] = (float)1 - i;
    b[i] = (float)i + 10;

    if (i % 100 == 0)
      a[i] = i;
  }

  CALL_KERNEL(gsumif, a);
  return 0;
}
