#include "if_loop_3.h"
#include "../integration_utils.h"
#include <stdlib.h>

int if_loop_3(in_int_t a[N], in_int_t b[N], in_int_t n) {
  int dist;
  int sum = 1000;
  for (int i = 0; i < n; i++) {
    dist = a[i] - b[i];
    if (dist >= 0)
      sum /= dist;
  }
  return sum;
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  for (int j = 0; j < N; ++j) {
    a[j] = rand() % 10;
    b[j] = a[j] + 1;
  }

  CALL_KERNEL(if_loop_3, a, b, N);
  return 0;
}
