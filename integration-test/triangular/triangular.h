//===- triangular.h - Triangular multiplication -------------------*- C -*-===//
//
// Declares the triangular kernel which multiplies the elements above a matrix
// diagonal's with a vector.
//
//===----------------------------------------------------------------------===//

typedef int in_int_t;
typedef int inout_int_t;

#define N 10

/// Multiplies the elements above a matrix diagonal's with the vector, updating
/// the matrix in the process.
void triangular(in_int_t x[N], in_int_t n, inout_int_t a[N][N]);
