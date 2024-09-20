//===- lu.c -LU-decomposition-------------------------------------*- C -*-===//
//
// lu.c: This file is adapted from Rosetta code
//
// Source: https://rosettacode.org/wiki/LU_decomposition#C
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "lu.h"
#include <math.h>

void lu(in_float_t A[N][N], inout_float_t L[N][N], inout_float_t U[N][N],
        inout_float_t P[N][N], in_float_t A_prime[N][N]) {

  // Pivot
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      P[i][j] = (i == j);
    }
  }

  for (int i = 0; i < N; i++) {
    int max_j = i;
    for (int j = i; j < N; j++) {
      if (fabs(A[j][i]) > fabs(A[max_j][i]))
        max_j = j;
    }
    if (max_j != i)
      for (int k = 0; k < N; k++) {
        int tmp = P[i][k];
        P[i][k] = P[max_j][k];
        P[max_j][k] = tmp;
      }
  }

  // A_prime = P * A
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++)
        A_prime[i][j] += P[i][k] * A[k][j];

  for (int i = 0; i < N; i++) {
    L[i][i] = 1;
  }

  // Final part
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      double s;
      if (j <= i) {
        s = 0;
        for (int k = 0; k < j; k++) {
          s += L[j][k] * U[k][i];
        }
        U[j][i] = A_prime[j][i] - s;
      }
      if (j >= i) {
        s = 0;
        for (int k = 0; k < i; k++) {
          s += L[j][k] * U[k][i];
        }
        L[j][i] = (A_prime[j][i] - s) / U[i][i];
      }
    }
}

int main() {
  in_float_t A[N][N] = {{1, 3, 5}, {2, 4, 7}, {1, 1, 0}};
  inout_float_t L[N][N];
  inout_float_t U[N][N];
  inout_float_t P[N][N];
  inout_float_t A_prime[N][N];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      L[i][j] = 0.0;
      U[i][j] = 0.0;
      A_prime[i][j] = 0.0;
    }
  }
  CALL_KERNEL(lu, A, L, U, P, A_prime);

#ifdef VERBOSE
  _PRINT(A);
  _PRINT(L);
  _PRINT(U);
  _PRINT(P);
#endif

  return 0;
}
