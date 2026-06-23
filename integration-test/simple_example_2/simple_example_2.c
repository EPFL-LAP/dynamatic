#include "simple_example_2.h"
#include "dynamatic/Integration.h"

void simple_example_2(inout_int_t a[N], int c) {
  int cond = c > 0;
  int i = 0;
  do {
    if (cond) {
      int x = c * 3;
      a[i] = x;
    }
    i++;
  } while (i < N);
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  int c;
  for (unsigned j = 0; j < N; ++j) {
    a[j] = j;
    b[j] = j;
  }

  CALL_KERNEL(simple_example_2, a, c);
  return 0;
}
