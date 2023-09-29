//===- bicg.h - BiCGSTAB (BiConjugate Gradient STABilized method) -*- C -*-===//
//
// Declares the bicg kernel which computes a biconjugate gradient between a
// matric and two vectors.
//
//===----------------------------------------------------------------------===//

#ifndef BICG_BICG_H
#define BICG_BICG_H

#define N 30

/// Computes the biconjugate gradient between a matruix and two vectors, and
/// stores the result in the two last arrays.
int bicg(int a[N][N], int s[N], int q[N], int p[N], int r[N]);

#endif // BICG_BICG_H
