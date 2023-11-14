//===- polyn_mult.c - Polynomial muliplication  -------------------*- C -*-===//
//
// Implements the polyn_mult kernel.
//
//===----------------------------------------------------------------------===//

#include "polyn_mult.h"
#include "../integration_utils.h"

unsigned polyn_mult(in_int_t a[N], in_int_t b[N], out_int_t out[N]) {
  unsigned p = 0;
  for (unsigned k = 0; k < N; k++) {
    out[k] = 0;
    unsigned i = 0;
    for (i = 1; i < N - k; i++)
      out[k] += a[k + i] * b[N - i];
    for (i = 0; i < k + 1; i++)
      out[k] += a[k - i] * b[i];
    p = i + k;
  }
  return p;
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  out_int_t out[N];

  for (int i = 0; i < N; i++) {
    out[i] = 0;
    a[i] = i % 10;
    b[i] = (N - i) % 10;
  }

  CALL_KERNEL(polyn_mult, a, b, out);
  return 0;
}
