//===- spmv.c -----------------------------------------------------*- C -*-===//
//
// sparse-matrix--dense-vector multiplication
//
// This benchmark might need to be revised for what it actually does.
//
//===----------------------------------------------------------------------===//

#include "spmv.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

int spmv(in_int_t n, inout_int_t row[11], inout_int_t col[10],
         inout_int_t val[10], inout_int_t vec[10], inout_int_t out[10]) {
  int s = 0;
  int e = 0;
  int tmp, cid;
  for (int i = 0; i < n; i++) {
    tmp = 0;
    s = row[i];
    e = row[i + 1];
    for (int c = s; c < e; c++) {
      cid = col[c];
      tmp += val[c] * vec[cid];
    }
    out[i] = tmp;
  }
  return s;
}

int main(void) {
  in_int_t n;
  // The row pointer in the CSR format has one extra element than n
  inout_int_t row[11];
  inout_int_t col[10];
  inout_int_t val[10];
  inout_int_t vec[10];
  inout_int_t out[10];

  n = 10;
  for (int i = 0; i < 10; i++) {
    row[i] = rand() % 10;
    col[i] = rand() % 5;
    val[i] = rand() % 10;
    vec[i] = rand() % 10;
    out[i] = rand() % 10;
  }
  row[n] = rand() % 10;
  CALL_KERNEL(spmv, n, row, col, val, vec, out);
}
