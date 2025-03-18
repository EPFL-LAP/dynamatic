//===- rouzbeh.c ---------------------------------------------*- C -*-===//

#include "dynamatic/Integration.h"
#include "rouzbeh.h"
#include <stdlib.h>

void rouzbeh(in_int_t iteration_store[2], in_int_t iteration_load[2],
  inout_float_t array[1000], in_int_t n) {
  
  
  float x = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < iteration_store[i]; j++){
      array[j] = x;
    }
    for (int k = 0; k < iteration_load[i]; k++){
      x = array[k];
    }
  }
}

int main(void) {
  in_int_t iteration_store[2] = {2, 2};
  in_int_t iteration_load[2] = {10, 10};
  inout_float_t array[1000];
  in_int_t n = 2;

  for (int i = 0; i < 1000; ++i) {
    array[i] = rand() % 1000;
  }


  CALL_KERNEL(rouzbeh, iteration_store, iteration_load, array, n);
  return 0;
}
