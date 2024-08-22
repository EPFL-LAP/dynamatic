#include "loop_array.h"
#include "dynamatic/Integration.h"
void loop_array(in_int_t n, in_int_t k, inout_int_t c[10]) {
  for (int i = 1; i < n; i++)
    c[i] = k + c[i - 1];
}

int main(void) {
  in_int_t n;
  in_int_t k;
  inout_int_t c[N];

  k = 25 % 10;
  n = 8;
  for (int j = 0; j < 10; ++j)
    c[j] = 0;

  CALL_KERNEL(loop_array, n, k, c);
  return 0;
}
