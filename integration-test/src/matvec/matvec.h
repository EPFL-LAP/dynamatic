//===- matvec.h - Multiply matrix to vector  ----------------------*- C -*-===//
//
// Declares the matvec kernel which multiplies a matrix by a vector.
//
//===----------------------------------------------------------------------===//

#ifndef MATVEC_MATVEC_H
#define MATVEC_MATVEC_H

#define N 100

/// Multiplies a matrix by a vector and store the result vector in the last
/// argument.
int matvec(int m[N][N], int v[N], int out[N]);

#endif // MATVEC_MATVEC_H
