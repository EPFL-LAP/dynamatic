#include "loop_add.h"
#include "dynamatic/Integration.h"

float loop_add(in_int_t a[N]) {
  float x = 2.0f;
  for (unsigned i = 0; i < N; ++i) {
    if (a[i] == 0)
      x = x * 1.25f;
  }
  return x;
}

int main(void) {
  in_int_t a[N];
  // Initialize a to [0, 1, 0, 1, ...]
  for (unsigned i = 0; i < N; ++i)
    a[i] = i % 2;
  CALL_KERNEL(loop_add, a);
  return 0;
}