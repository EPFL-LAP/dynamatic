//===- fir.h - Computes FIR of two integer arrays -----------------*- C -*-===//
//
// Declares the fir kernel which computes a finite impulse response (FIR)
// between two discrete signals.
//
//===----------------------------------------------------------------------===//

#ifndef FIR_FIR_H
#define FIR_FIR_H

#define N 1000
#define N_DEC 999 // = N - 1

/// Computes the finite impulse response between two arrays.
int fir(int di[N], int idx[N]);

#endif // FIR_FIR_H
