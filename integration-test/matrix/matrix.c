//===- matrix.c - Matrix multiplication ---------------------------*- C -*-===//
//
// Declares the matrix kernel which computes a matrix multiplication.
//
//===----------------------------------------------------------------------===//

#include "matrix.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void matrix(in_int_t inA[A_ROWS][A_COLS], in_int_t inB[A_COLS][B_COLS],
            out_int_t outC[A_ROWS][B_COLS]) {
  for (unsigned i = 0; i < A_ROWS; i++) {
    for (unsigned j = 0; j < B_COLS; j++) {
      int sumMult = 0;
      for (unsigned k = 0; k < A_COLS; k++) {
        sumMult += inA[i][k] * inB[k][j];
      }
      outC[i][j] = sumMult;
    }
  }
}

int main(void) {
  in_int_t inA[A_ROWS][A_COLS];
  in_int_t inB[B_ROWS][B_COLS];
  out_int_t outC[A_ROWS][B_COLS];

  srand(13);
  for (int y = 0; y < A_ROWS; ++y) {
    for (int x = 0; x < A_COLS; ++x) {
      inA[y][x] = rand() % 10;
    }
  }
  for (int y = 0; y < B_ROWS; ++y) {
    for (int x = 0; x < B_COLS; ++x) {
      inB[y][x] = rand() % 10;
    }
  }

  CALL_KERNEL(matrix, inA, inB, outC);
  return 0;
}
