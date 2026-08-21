//===- mat_mul.c - Computes FIR of two integer arrays -----------------*- C
//-*-===//
//
// Declares the mat_mul kernel which computes a finite impulse response (FIR)
// between two discrete signals.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "mat_mul.h"
#include "stdlib.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

/*
extern int fpsa_mat_mul(in_int_t A[ROWS][W], in_int_t B[W][COLS],
                        in_int_t C[ROWS][COLS], int SA_ROWS, int SA_COLS);

void mat_mul2(in_int_t A[ROWS][W], in_int_t B[W][COLS],
              in_int_t C[ROWS][COLS]) {
  int SA_ROWS = 4, SA_COLS = 4;
  fpsa_mat_mul(A, B, C, SA_ROWS, SA_COLS);
}
*/
void mat_mul(in_int_t aggregatedMemref0[ROWS * W + COLS * W + ROWS * COLS]);

void mat_mul(in_int_t aggregatedMemref0[ROWS * W + COLS * W + ROWS * COLS]) {
  for (unsigned i = 0; i < ROWS; i++) {
    for (unsigned j = 0; j < COLS; j++) {
      in_int_t sum = 0;
      for (unsigned k = 0; k < W; k++) {
        sum += aggregatedMemref0[i * W + k] *
               aggregatedMemref0[ROWS * W + j * W + k];
      }
      aggregatedMemref0[ROWS * W + COLS * W + i * COLS + j] = sum;
    }
  }
}

int main(void) {

  in_int_t A[ROWS][W];
  in_int_t B[COLS][W];
  in_int_t C[ROWS][COLS];
  in_int_t aggregatedMemref0[ROWS * W + COLS * W + ROWS * COLS];
  for (unsigned i = 0; i < ROWS; i++)
    for (unsigned j = 0; j < W; j++) {
      A[i][j] = rand() % 50;
      aggregatedMemref0[i * W + j] = A[i][j];
    }

  for (unsigned i = 0; i < COLS; i++)
    for (unsigned j = 0; j < W; j++) {
      B[i][j] = rand() % 50;
      aggregatedMemref0[ROWS * W + i * W + j] = B[i][j];
    }
  CALL_KERNEL(mat_mul, aggregatedMemref0);
  return 0;
}
