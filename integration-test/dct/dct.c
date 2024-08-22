//===- dct.c ------------------------------------------------------*- C -*-===//
//
// A naive implementation of discrete cosine tranform
//
//
//===----------------------------------------------------------------------===//

#include "dct.h"
#include "dynamatic/Integration.h"
#include <math.h>

void dct(out_double_t output_matrix[N][M], in_double_t input_matrix[N][M]) {

  int i, j, u, v;
  for (u = 0; u < N; ++u) {
    for (v = 0; v < M; ++v) {
      output_matrix[u][v] = 0.0;
      for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
          output_matrix[u][v] += input_matrix[i][j] *
                                 cos(M_PI / (((double)N) * (i + 0.5)) * u) *
                                 cos(M_PI / (((double)M) * (j + 0.5)) * v);
        }
      }
    }
  }
}

#define AMOUNT_OF_TEST 1

int main(void) {
  in_double_t input_matrix[N][M];

  out_double_t output_matrix[N][M];

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      input_matrix[i][j] = (double)(0.0);
      output_matrix[i][j] = (double)(0.0);
    }
  }
  CALL_KERNEL(dct, input_matrix, output_matrix);
}
