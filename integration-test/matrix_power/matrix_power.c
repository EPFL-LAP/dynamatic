#include "matrix_power.h"
#include "../integration_utils.h"
#include <stdlib.h>

void matrix_power(inout_int_t x[N][N], in_int_t row[N], in_int_t col[N],
                  in_int_t a[N]) {
  for (unsigned k = 1; k < N; k++) {
    for (unsigned p = 0; p < N; p++)
      x[k][row[p]] += a[p] * x[k - 1][col[p]];
  }
}

int main(void) {
  inout_int_t mat[N][N];
  in_int_t row[N];
  in_int_t col[N];
  in_int_t a[N];

  for (unsigned y = 0; y < N; ++y) {
    col[y] = rand() % N;
    row[y] = rand() % N;
    a[y] = rand();
    for (unsigned x = 0; x < N; ++x)
      mat[y][x] = 0;
  }

  CALL_KERNEL(matrix_power, mat, row, col, a);
  return 0;
}
