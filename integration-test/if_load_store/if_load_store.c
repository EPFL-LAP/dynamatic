//===- loop_load.c ---------------------------------------------*- C -*-===//

#include "dynamatic/Integration.h"
#include "if_load_store.h"
#include <stdlib.h>

void if_load_store(in_int_t a[1000], in_int_t b[1000], in_int_t n) {
    if (a[5] < 10){
      b[17] = a[10] + 5;
    } 
    a[10] = b[19];
}

int main(void) {
  in_int_t a[1000];
  in_int_t b[1000];
  in_int_t n;

  n = 1000;
  for (int i = 0; i < 1000; ++i) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  CALL_KERNEL(if_load_store, a, b, n);
  return 0;
}
