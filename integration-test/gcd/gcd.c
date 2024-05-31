//===- gcd.c - Computes GCD of two integers  ----------------------*- C -*-===//
//
// Implements the gcd kernel (https://en.algorithmica.org/hpc/algorithms/gcd/).
//
//===----------------------------------------------------------------------===//

#include "gcd.h"
#include "dynamatic/Integration.h"

int gcd(in_int_t a, in_int_t b) {
  if (a == 0)
    return b;
  if (b == 0)
    return a;

  // Find the greatest power of 2 that divides both a and b
  unsigned k = 0;
  while (((a | b) & 1) == 0) {
    a >>= 1;
    b >>= 1;
    k++;
  }

  // a := __builtin_ctz(a)
  while (a > 0 && (a & 1) == 0)
    a >>= 1;
  // b := __builtin_ctz(b)
  while (b > 0 && (b & 1) == 0)
    b >>= 1;

  while (a != 0) {
    in_int_t diff = a - b;
    // b := min(a, b);
    if (a < b)
      b = a;
    // a := abs(diff)
    if (diff >= 0)
      a = diff;
    else
      a = -diff;
    // a := __builtin_ctz(a)
    while (a > 0 && (a & 1) == 0)
      a >>= 1;
  }

  return b << k;
}

int main(void) {
  in_int_t a = 7966496;
  in_int_t b = 314080416;
  CALL_KERNEL(gcd, a, b);
  return 0;
}
