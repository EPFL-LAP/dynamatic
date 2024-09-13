#include "dynamatic/Integration.h"
#include "while_loop_2.h"
#include <stdlib.h>

void while_loop_2(inout_int_t a[1000]) {
  int i = 0;
  int bound = 1000;
  int sum = 0;

  while (a[i] != 0) {
    i++;
  }
  a[0] = i;
}

int main(void) {
  inout_int_t a[1000];

  for (int i = 0; i < 1000; ++i) {
    a[i] = rand() % 100;
  }
  a[999] = 0;

  CALL_KERNEL(while_loop_2, a);
}
