//===- matvec.h - Multiply matrix to vector  ----------------------*- C -*-===//
//
// Declares the matvec kernel which multiplies a matrix by a vector.
//
//===----------------------------------------------------------------------===//

#ifndef MATVEC_MATVEC_H
#define MATVEC_MATVEC_H

#define N 100

typedef int in_int_t;
typedef int out_int_t;

/// Multiplies a matrix by a vector and store the result vector in the last
/// argument.
int matvec(in_int_t m[N][N], in_int_t v[N], out_int_t out[N]);

#endif // MATVEC_MATVEC_H
