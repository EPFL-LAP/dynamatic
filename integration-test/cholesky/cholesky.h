//===- cholesky.h -  Cholesky Decomposition -----------------------*- C -*-===//
//
//===----------------------------------------------------------------------===//

typedef float in_float_t;
typedef float out_float_t;
typedef float inout_float_t;

#define N 10

/// Multiplies two matrices and stores the multiplication's result in the last
/// argument.
void cholesky(inout_float_t A[N][N], out_float_t R[N][N]);
