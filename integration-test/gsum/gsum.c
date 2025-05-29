//===- gsum.c - Testing long-latency--loop-carried dependency --*- C -*-===//
//
// Implements a kernel with long-latency--loop-carried dependency in the loop
// body
//
// Author: Jianyi Cheng
// https://zenodo.org/record/3561115
//
//===------------------------------------------------------------------===//

#include "gsum.h"
#include "dynamatic/Integration.h"

float gsum(in_float_t a[N]) {
  int i;
  float d;
  float s = 0.0;

  for (i = 0; i < N; i++) {
    d = a[i];
    if (d >= 0)
      // An if condition in the loop causes irregular computation.  Static
      // scheduler reserves time slot for each iteration causing unnecessary
      // pipeline stalls.

      s += (((((d + (float)0.64) * d + (float)0.7) * d + (float)0.21) * d +
             (float)0.33) *
            d);
  }
  return s;
}

int main(void) {
  in_float_t a[N];
  in_float_t b[N];

  for (int i = 0; i < N; ++i) {
    a[i] = (float)1 - i;
    b[i] = (float)i + 10;

    if (i % 100 == 0)
      a[i] = i;
  }

  CALL_KERNEL(gsum, a);
  return 0;
}
