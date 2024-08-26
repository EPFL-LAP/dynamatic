//===- get_tanh.c - Long-latency--loop-carried dependency -----*- C -*-===//
//
// Implements a kernel with long-latency--loop-carried dependency in the loop
// body
//
// Author: Jianyi Cheng
// https://zenodo.org/record/3561115
//
//===------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "get_tanh.h"
#include <stdlib.h>

void get_tanh(inout_float_t A[1000], in_int_t addr[1000]) {
  int i;
  float result, beta;

  for (i = 0; i < 1000; i++) {

    int address = addr[i];
    beta = A[address];

    if (beta >= (float)1.0) {
      result = (float)1.0;
    } else {
      result =
          ((beta * beta + (float)19.52381) * beta * beta + (float)3.704762) *
          beta;
    }
    A[address] = result;
  }
}

int main(void) {
  inout_float_t A[1000];
  in_int_t addr[1000];

  for (int i = 0; i < 1000; ++i) {
    A[i] = (float)i;
    addr[i] = i;

    if (i % 100 == 0)
      A[i] = 0;
  }

  CALL_KERNEL(get_tanh, A, addr);
}
