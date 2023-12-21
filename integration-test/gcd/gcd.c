//===- gcd.c - Computes GCD of two integers  ----------------------*- C -*-===//
//
// Implements the gcd kernel.
//
//===----------------------------------------------------------------------===//

#include "gcd.h"
#include "../integration_utils.h"

int gcd(in_int_t a, in_int_t b) {
  // Finding K, where K is the greatest power of 2 that divides both in0 and
  // in1. for (int k = 0; ((in0 | in1) & 1) == 0; ++k)
  unsigned k = 0;
  while (((a | b) & 1) == 0) {
    a >>= 1;
    b >>= 1;
    k++;
  }

  // Dividing in0 by 2 until in0 becomes odd
  while ((a & 1) == 0)
    a >>= 1;

  // From here on, 'in0' is always odd.
  // If in1 is even, remove all factor of 2 in in1
  while ((b & 1) == 0)
    b >>= 1;

  // Now in0 and in1 are both odd. Swap if necessary so in0 <= in1, then set in1
  // = in1 - in0 (which is even).
  while (b > 0 && ((b & 1) == 0)) {
    b = b - a;
  }

  // Restore common factors of 2
  return a << k;
}

int main(void) {
  CALL_KERNEL(gcd, 7966496, 314080416);
  return 0;
}
