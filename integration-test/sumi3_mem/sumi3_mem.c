#include "sumi3_mem.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

int sumi3_mem(in_int_t a[N]) {
  int sum = 0;
  for (unsigned i = 0; i < N; i++) {
    int x = a[i];
    sum += x * x * x;
  }
  return sum;
}

int main(void) {
  in_int_t a[N];
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(sumi3_mem, a);
  return 0;
}
