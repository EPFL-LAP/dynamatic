#ifndef LU_LU_H
#define LU_LU_H
typedef float in_float_t;
typedef float inout_float_t;

#define N 3

void lu(in_float_t A[N][N], inout_float_t L[N][N], inout_float_t U[N][N],
        inout_float_t P[N][N], in_float_t A_prime[N][N]);

#ifdef VERBOSE
#include <stdio.h>
#define _PRINT(X)                                                              \
  for (int i = 0; i < N; i++) {                                                \
    for (int j = 0; j < N; j++) {                                              \
      printf("%8.4g", X[i][j]);                                                \
      printf(" ");                                                             \
    }                                                                          \
    printf("\n");                                                              \
  }                                                                            \
  printf("\n")
#endif

#endif // LU_LU_H
