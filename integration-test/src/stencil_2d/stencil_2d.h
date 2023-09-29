//===- stencil_2d.h - Grid-based computation ----------------------*- C -*-===//
//
// Declares the stencil_2d kernel which performs a grid-based computation.
//
//===----------------------------------------------------------------------===//

#ifndef STENCIL_2D_STENCIL_2D_H
#define STENCIL_2D_STENCIL_2D_H

#define N 900
#define M 10

/// Performs a grid-based computation based on the data contained in the first
/// two arrays, and stores results in the last array.
// NOLINTNEXTLINE(readability-identifier-naming)
int stencil_2d(int orig[N], int filter[M], int sol[N]);

#endif // STENCIL_2D_STENCIL_2D_H
