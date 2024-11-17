#include "nested_loop.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

void nested_loop(in_int_t a[N], in_int_t b[N], inout_int_t c[N]) {
  for(int j = 0; j < 2; j++){
    int i = 0;
    int bound = 1000;
    int sum = 0;
    do {
      sum = a[i] * b[i];
      c[i + j * 400] = sum;
      i++;
    } while (sum < bound);
  }
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  inout_int_t c[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 5;
    b[j] = j;
    c[j] = 0;
  }

  CALL_KERNEL(nested_loop, a, b, c);
  return 0;
}
