//===- stencil_2d.h - Grid-based computation ----------------------*- C -*-===//
//
// Declares the stencil_2d kernel which performs a grid-based computation.
//
//===----------------------------------------------------------------------===//

#ifndef STENCIL_2D_STENCIL_2D_H
#define STENCIL_2D_STENCIL_2D_H

#define N 900
#define M 10

typedef int in_int_t;
typedef int out_int_t;

/// Performs a grid-based computation based on the data contained in the first
/// two arrays, and stores results in the last array.
int stencil_2d(in_int_t orig[N], in_int_t filter[M], out_int_t sol[N]);

#endif // STENCIL_2D_STENCIL_2D_H
