//===- ii_nested_int.h - Nested loop with an II=1 recurrence ------*- C -*-===//
//
// Declares the ii_nested_int kernel, whose inner loop carries an integer
// accumulation: a combinational recurrence that pipelines at II=1.
//
//===----------------------------------------------------------------------===//

#ifndef II_NESTED_INT_II_NESTED_INT_H
#define II_NESTED_INT_II_NESTED_INT_H

#define N 8
#define M 30

typedef int in_int_t;
typedef int out_int_t;

/// Sums each row of a matrix into a vector of per-row sums.
void ii_nested_int(in_int_t a[N][M], out_int_t sums[N]);

#endif // II_NESTED_INT_II_NESTED_INT_H
