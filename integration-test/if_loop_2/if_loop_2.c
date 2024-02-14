#include "if_loop_2.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

int if_loop_2(in_int_t a[N], in_int_t n) {
  int sum = 0;
  for (int i = 0; i < n; i++) {
    int tmp = a[i];
    if (tmp > 10)
      sum += tmp;
  }
  return sum;
}

int main(void) {
  in_int_t a[N];
  in_int_t n = N;
  for (int j = 0; j < N; ++j)
    a[j] = rand() % N;

  CALL_KERNEL(if_loop_2, a, n);
  return 0;
}
