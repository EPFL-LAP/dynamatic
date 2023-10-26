//===- gcd.h - Computes GCD of two integers  ----------------------*- C -*-===//
//
// Declares the gcd kernel which computes the greatest common denominator (GCD)
// between two integers using stein's algorithm.
//
//===----------------------------------------------------------------------===//

#ifndef GCD_GCD_H
#define GCD_GCD_H

typedef int in_int_t;

/// Computes the GCD between two integers.
int gcd(in_int_t a, in_int_t b);

#endif // GCD_GCD_H
