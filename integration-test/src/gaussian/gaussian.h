//===- gaussian.h - Gaussian computation --------------------------*- C -*-===//
//
// Declares the gaussian kernel which computes a gaussian between a vector and
// matrix.
//
//===----------------------------------------------------------------------===//

#ifndef GAUSSIAN_GAUSSIAN_H
#define GAUSSIAN_GAUSSIAN_H

#define N 20
#define N_DEC 19     // = N - 1
#define N_DEC_DEC 18 // = N - 2

typedef int in_int_t;
typedef int inout_int_t;

/// Computes the gaussian between a vector and matrix.
unsigned gaussian(in_int_t c[N], inout_int_t a[N][N]);

#endif // GAUSSIAN_GAUSSIAN_H
