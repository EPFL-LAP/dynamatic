// clang-format off
#include "sparse.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

float sparse(in_float_t a[N], in_float_t x[N]) {
  float sum = 0.0f;
  int i = 0;
  float mul;
  bool loopAgain;
  do {
    mul = a[i] * x[i];
    sum += mul;
    i++;
    loopAgain = sum >= 0.0f;
    #pragma DYN speculate variable=loopAgain max_predictions=2 style=standard
  } while (loopAgain);
  return sum;
}

int main(void) {
  in_float_t a[N];
  in_float_t x[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    // a[j] = 1.2f - j;
    // x[j] = j;
    a[j] = 40.0f - j;
    x[j] = 1.0f;
  }

  CALL_KERNEL(sparse, a, x);
  return 0;
}
