#include "finite_loop.h"
#include "dynamatic/Integration.h"

void finite_loop(inout_float_t a[N], inout_int_t b[N]) {
  int x = 0;
  for (unsigned i = 1; i < N; ++i) {
    a[i] = a[i-1] * -1.0+6.0;
    // x = a[i]*4 + 8 - x;
    // b[i] = i;
  } 
}

int main(void) {
  in_float_t a[N];
  in_int_t b[N];
  for (unsigned j = 0; j < N; ++j) {
    a[j] = 1.0;
    b[j] = j;
  }

  CALL_KERNEL(finite_loop, a, b);
  return 0;
}
