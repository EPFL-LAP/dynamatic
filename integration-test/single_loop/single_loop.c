// clang-format off
#include "single_loop.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

void single_loop(in_int_t a[N], in_int_t b[N], inout_int_t c[N]) {
  int i = 0;
  int bound = 1000;
  int sum = 0;
  bool loopAgain;
  do {
    sum = a[i] * b[i];

    c[i] = sum;
    i++;
    loopAgain = sum < bound;

#pragma DYN speculate variable = loopAgain max_predictions = 6 style = standard
  } while (loopAgain);
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  inout_int_t c[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 2;
    b[j] = j;
    c[j] = 0;
  }

  CALL_KERNEL(single_loop, a, b, c);
  return 0;
}
