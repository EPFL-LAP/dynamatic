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

typedef int in_int_t;

/// Computes the finite impulse response between two arrays.
int fir(in_int_t di[N], in_int_t idx[N]);

#endif // FIR_FIR_H
