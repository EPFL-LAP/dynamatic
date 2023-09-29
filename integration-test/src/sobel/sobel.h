//===- sobel.h - Sobel filter -------------------------------------*- C -*-===//
//
// Declares the sobel kernel which computes a Sobel filter.
//
//===----------------------------------------------------------------------===//

#ifndef SOBEL_SOBEL_H
#define SOBEL_SOBEL_H

#define N 256

/// Computes the Sobel filter of the first argument with two 3x3 kernels passed
/// as second and third arguments and stores the result in the last argument.
int sobel(int in[N], int gX[9], int gY[9], int out[N]);

#endif // SOBEL_SOBEL_H
