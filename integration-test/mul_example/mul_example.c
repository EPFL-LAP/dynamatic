#include "mul_example.h"
#include "../integration_utils.h"

void mul_example(inout_int_t a[N]) {
  for (unsigned i = 0; i < N; ++i) {
    int x = a[i];
    a[i] = x * x * x;
  }
}

int main(void) {
  in_int_t a[N];
  for (unsigned j = 0; j < N; ++j)
    a[j] = j;

  CALL_KERNEL(mul_example, a);
  return 0;
}