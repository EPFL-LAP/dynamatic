//===- gaussian.h - Gaussian computation --------------------------*- C -*-===//
//
// Declares the gaussian kernel which computes a gaussian between a vector and
// matrix.
//
//===----------------------------------------------------------------------===//

#ifndef GAUSSIAN_GAUSSIAN_H
#define GAUSSIAN_GAUSSIAN_H

#include <cstddef>

#define N 20
#define N_DEC 19     // = N - 1
#define N_DEC_DEC 18 // = N - 2

/// Computes the gaussian between a vector and matrix.
size_t gaussian(int c[N], int a[N][N]);

#endif // GAUSSIAN_GAUSSIAN_H
