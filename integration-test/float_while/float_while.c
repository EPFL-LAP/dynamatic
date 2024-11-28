#include "float_while.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

void float_while(in_float_t a[N], in_float_t x[N], inout_float_t b[N]) {
  int i = 0;
  float mul = 0.0f;
  while (mul >= 0.0f) {
    mul = a[i] * x[i];
    if (mul >= 5.0f) {
      break;
    }
    b[i] = mul;
    i++;
  }
}

int main(void) {
  in_float_t a[N];
  inout_float_t b[N];
  in_float_t x[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 40.0f - j;
    x[j] = 1.0f;
  }

  CALL_KERNEL(float_while, a, x, b);
  return 0;
}
