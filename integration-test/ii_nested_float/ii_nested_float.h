//===- ii_nested_float.h - Nested loop with a slow recurrence -----*- C -*-===//
//
// Declares the ii_nested_float kernel, whose inner loop carries a
// floating-point accumulation: a recurrence of several cycles, run for only a
// few iterations per activation of the frequently re-entered inner loop.
//
//===----------------------------------------------------------------------===//

#ifndef II_NESTED_FLOAT_II_NESTED_FLOAT_H
#define II_NESTED_FLOAT_II_NESTED_FLOAT_H

#define N 20
#define M 5

typedef float in_float_t;
typedef float out_float_t;

/// Sums each row of a matrix into a vector of per-row sums.
void ii_nested_float(in_float_t a[N][M], out_float_t sums[N]);

#endif // II_NESTED_FLOAT_II_NESTED_FLOAT_H
