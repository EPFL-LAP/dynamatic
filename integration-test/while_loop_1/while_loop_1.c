#include "dynamatic/Integration.h"
#include "while_loop_1.h"
#include <stdlib.h>

void while_loop_1(inout_int_t a[1000], in_int_t b[1000]) {
  int i = 0;
  int bound = 1000;
  int sum = 0;

  while (sum < bound) {

    sum = a[i] + b[i];
    a[i] = sum;
    i++;
  }
}

int main(void) {
  inout_int_t a[1000];
  inout_int_t b[1000];

  for (int i = 0; i < 1000; ++i) {
    a[i] = rand() % 10;
    b[i] = rand() % 10;
  }
  a[900] = 600;
  b[900] = 600;

  CALL_KERNEL(while_loop_1, a, b);
  return 0;
}
