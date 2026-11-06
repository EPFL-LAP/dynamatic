#include "sparse.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

float sparse(in_float_t a[N], in_float_t x[N]) {
  float sum = 0.0f;
  int i = 0;
  float mul;
  for (i = 0; i < N; i++) {
    mul = 12.0f * x[i];
    sum += mul;
  }
  return sum;
}

int main(void) {
  in_float_t a[N];
  in_float_t x[N];

  for (int j = 0; j < N; ++j) {
    a[j] = 40.0f - j;
    if (j < 0) {
      x[j] = 7.0f;
    } else {
      x[j] = 0.0f;
    }
  }

  CALL_KERNEL(sparse, a, x);
  return 0;
}
