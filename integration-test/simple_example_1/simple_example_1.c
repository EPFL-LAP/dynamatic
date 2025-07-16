#include "dynamatic/Integration.h"
#include "simple_example_1.h"

void simple_example_1(inout_int_t a[N]) {
  // int x = a[0];
  // a[0] = 42; // Set the first element to 42
  // for (unsigned i = 0; i < N; ++i)
  //   a[i] += 1;

  for (unsigned i = 0; i < N; ++i) {
    a[a[i]] = a[i] + a[i] * 5; // Increment each element by 1
  }
  
}

int main(void) {
  in_int_t a[N];
  for (unsigned j = 0; j < N; ++j)
    a[j] = j;

  CALL_KERNEL(simple_example_1, a);
  return 0;
}
