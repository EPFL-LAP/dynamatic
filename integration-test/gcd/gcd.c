//===- gcd.c - Computes GCD of two integers  ----------------------*- C -*-===//
//
// Implements the gcd kernel (https://en.algorithmica.org/hpc/algorithms/gcd/).
//
//===----------------------------------------------------------------------===//

#include "gcd.h"
#include "dynamatic/Integration.h"

#define M 200
#define N 12
#define P 15

int gcd(in_int_t a, in_int_t b) {
  int i = a;
  int j = b;
  do {
    j = i + a;
    i = j + b;
  } while (i < 10);
  return i * j / a;
}

int main(void) {
  in_int_t a = 7966496;
  in_int_t b = 314080416;
  CALL_KERNEL(gcd, a, b);
  return 0;
}
