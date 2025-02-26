#include "while_loop_3.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void while_loop_3(out_int_t a[ARRAY_SIZE], in_int_t bound) {
  int i = 0;
  int sum = 0;

  while (i * i < bound) {
    i++;
  }
  a[0] = i;
}

int main(void) {
  out_int_t a[ARRAY_SIZE];
  in_int_t bound;

  bound = 1;
  for (int j = 0; j < ARRAY_SIZE; j++) {
    a[j] = 0;
  }

  CALL_KERNEL(while_loop_3, a, bound);
}
