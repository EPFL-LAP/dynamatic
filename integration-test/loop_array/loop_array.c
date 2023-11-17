#include "loop_array.h"
#include "../integration_utils.h"
#include <stdlib.h>

void loop_array(in_int_t n, in_int_t k, inout_int_t c[10]) {
  for (int i = 1; i < n; i++)
    c[i] = k + c[i - 1];
}

int main(void) {
  in_int_t k;
  in_int_t n;
  inout_int_t c[N];

  srand(13);
  k = rand() % 10;
  n = rand() % 10;
  for (int j = 0; j < 10; ++j)
    c[j] = 0;

  CALL_KERNEL(loop_array, n, k, c);
  return 0;
}
