#include "sparse.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

float sparse(in_float_t a[N], in_float_t x[N]) {
  float sum = 0.0f;
  int i = 0;
  float mul;
  while (sum >= 0.0f) {
    mul = a[i] * x[i];
    sum += mul;
    i++;
  };
  return sum;
}

int main(void) {
  in_float_t a[N];
  in_float_t x[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    // a[j] = 1.2f - j;
    // x[j] = j;
    a[j] = 40.0f - j;
    x[j] = 1.0f;
  }

  CALL_KERNEL(sparse, a, x);
  return 0;
}
