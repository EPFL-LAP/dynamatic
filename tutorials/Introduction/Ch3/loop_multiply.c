#include "loop_multiply.h"
#include "dynamatic/Integration.h"

unsigned loopMultiply(in_int_t a[N]) {
  unsigned x = 2;
  for (unsigned i = 0; i < N; ++i) {
    if (a[i] == 0)
      x = x * x;
  }
  return x;
}

int main(void) {
  in_int_t a[N];
  // Initialize a to [0, 1, 0, 1, ...]
  for (unsigned i = 0; i < N; ++i)
    a[i] = i % 2;
  CALL_KERNEL(loopMultiply, a);
  return 0;
}
