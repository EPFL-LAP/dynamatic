#include "simple_example.h"
#include "../integration_utils.h"

void simple_example(inout_int_t a[N]) {
  int x = 0;
  for (unsigned i = 0; i < N; ++i)
    x++;
  a[0] = x;
}

int main(void) {
  in_int_t a[N];
  for (unsigned j = 0; j < N; ++j)
    a[j] = j;

  CALL_KERNEL(simple_example, a);
  return 0;
}