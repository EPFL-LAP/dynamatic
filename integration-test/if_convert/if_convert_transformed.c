#include "dynamatic/Integration.h"
#include "if_convert.h"
#include "stdbool.h"
#include "stdlib.h"

void if_convert(in_int_t a[N], inout_int_t b[N]) {
  int i = 1;
  do {
    int pred_i = i;
    do {
      int tmp = a[pred_i];

      // Manual CSE
      int plus_1 = pred_i + 1;

      int true_addition = pred_i * tmp < 10000 ? 2 : 1;
      i = pred_i + true_addition;
      b[i] = 1;

      // Prediction
      pred_i = plus_1;
    } while (pred_i < N2 && pred_i == i);
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
