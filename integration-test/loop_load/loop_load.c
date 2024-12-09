//===- loop_load.c ---------------------------------------------*- C -*-===//

#include "dynamatic/Integration.h"
#include "loop_load.h"
#include <stdlib.h>

void loop_load(in_int_t array[1000], in_int_t n) {
  int i, m;
  for (i = 0; i < n; ++i) {
    m = array[i];
    if (m > 60)
      break;
  }
  array[i-1] = 10;
}

int main(void) {
  in_int_t array[1000];
  in_int_t n;

  n = 1000;
  for (int i = 0; i < 1000; ++i) {
    array[i] = rand() % 100;
  }

  CALL_KERNEL(loop_load, array, n);
  return 0;
}
