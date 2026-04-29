#include "finite_loop.h"
#include "dynamatic/Integration.h"

void finite_loop(inout_int_t a[N], inout_int_t b[N]) {
  int x = 0;
  for (unsigned i = 1; i < N; ++i) {
    a[i] = a[i-1] *2+2;
    // x = a[i] + x;
    b[i] = i;
  } 
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  for (unsigned j = 0; j < N; ++j) {
    a[j] = -2;
    b[j] = j;
  }

  CALL_KERNEL(finite_loop, a, b);
  return 0;
}
