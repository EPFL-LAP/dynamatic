// clang-format off
#include "if_convert.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

void if_convert(in_int_t a[N], inout_int_t b[N]) {
  int i = 1;
  do {
    int tmp = a[i];
    bool ifPred = i * tmp < 10000;
    #pragma DYN speculate variable=ifPred max_predictions=7 style=standard
    if (ifPred) {
      i++;
    }
    i++;
    b[i] = 1;
  } while (i < N2);
}

int main(void) {
  in_int_t a[N];
  inout_int_t b[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = rand() % 5000;
    b[j] = 0;
  }

  CALL_KERNEL(if_convert, a, b);
  return 0;
}
