#include "pivot.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void pivot(inout_int_t x[N], in_int_t a[N], in_int_t n, in_int_t k) {
  int tmp = 0;
  for (int i = k + 1; i <= n; ++i) {
    x[k] = tmp;
    tmp = x[k] - a[i] * x[i];
  }
}

int main(void) {
  inout_int_t x[N];
  in_int_t a[N];
  in_int_t n = 100;
  in_int_t k = 2;

  for (unsigned j = 0; j < N; ++j) {
    x[j] = rand() % 100;
    a[j] = rand() % 100;
  }

  CALL_KERNEL(pivot, x, a, n, k);
  return 0;
}
