//===- sobel.h - Sobel filter -------------------------------------*- C -*-===//
//
// Declares the sobel kernel which computes a Sobel filter.
//
//===----------------------------------------------------------------------===//

#ifndef SOBEL_SOBEL_H
#define SOBEL_SOBEL_H

#define N 256

typedef int in_int_t;
typedef int out_int_t;

/// Computes the Sobel filter of the first argument with two 3x3 kernels passed
/// as second and third arguments and stores the result in the last argument.
int sobel(in_int_t in[N], in_int_t gX[9], in_int_t gY[9], out_int_t out[N]);

#endif // SOBEL_SOBEL_H
