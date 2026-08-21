//===- mat_mul.c - Computes FIR of two integer arrays -----------------*- C
//-*-===//
//
// Declares the mat_mul kernel which computes a finite impulse response (FIR)
// between two discrete signals.
//
//===----------------------------------------------------------------------===//

#include "mat_mul.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

extern int fpsa_mat_mul(in_int_t A[ROWS][W], in_int_t B[W][COLS],
                        in_int_t C[ROWS][COLS], int SA_ROWS, int SA_COLS);

void mat_mul(in_int_t A[ROWS][W], in_int_t B[W][COLS], in_int_t C[ROWS][COLS]);

void mat_mul(in_int_t A[ROWS][W], in_int_t B[W][COLS], in_int_t C[ROWS][COLS]) {
  int SA_ROWS = 4, SA_COLS = 4;
  fpsa_mat_mul(A, B, C, SA_ROWS, SA_COLS);
}

int main(void) {

  in_int_t A[ROWS][W];
  in_int_t B[W][COLS];
  in_int_t C[ROWS][COLS];
  for (unsigned i = 0; i < ROWS; i++)
    for (unsigned j = 0; j < W; j++)
      A[i][j] = rand() % 256;
  for (unsigned i = 0; i < W; i++)
    for (unsigned j = 0; j < COLS; j++)
      B[i][j] = rand() % 256;
  CALL_KERNEL(mat_mul, A, B, C);
  return 0;
}
