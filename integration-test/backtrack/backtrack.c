//===- backtrack.c ------------------------------------------------*- C -*-===//
//
// Remark: This is one of the benchmarks in FPGA '19
//
// TODO: find the source of this implementation
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include <stdlib.h>
int backtrack(float best[1000], float cost[1000]) {
  int i;
  for (i = 0; i < 1000; i++) {
    float temp = best[i] + cost[i];
    float x = 5.0;
    if ((1000 - temp) <= x * (temp))
      break;
  }
  return i;
}

int main(void) {
  float best[1000];
  float cost[1000];
  for (int i = 0; i < 1000; ++i) {
    best[i] = rand() % 6;
    cost[i] = rand() % 6;
  }
  CALL_KERNEL(backtrack, best, cost);
}
