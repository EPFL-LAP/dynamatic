//===- gemver.h - Vector multiplication and matrix addition -------*- C -*-===//
//
// Declares the gemver kernel which computes some vector multiplications and
// matrix additions.
//
//===----------------------------------------------------------------------===//

typedef int in_int_t;
typedef int inout_int_t;

#define N 30

/// Performs some vector multiplications and matrix additions between all
/// arguments and stores their results in the last three arguments.
void gemver(in_int_t alpha, in_int_t beta, in_int_t u1[N], in_int_t v1[N],
            in_int_t u2[N], in_int_t v2[N], in_int_t y[N], in_int_t z[N],
            inout_int_t a[N][N], inout_int_t w[N], inout_int_t x[N]);