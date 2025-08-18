#include "dynamatic/Integration.h"
#include "if_convert.h"
#include "stdbool.h"
#include "stdlib.h"

void if_convert(in_int_t a[N], inout_int_t b[N]) {
  int i = 1;
  while (i < N2) {
    int pred_i = i;
    while (pred_i < N2 && pred_i == i) {
      int tmp = a[pred_i];
      if (pred_i * tmp < 10000) {
        i = pred_i + 2;
      } else {
        i = pred_i + 1;
      }
      b[i] = 1;

      // Prediction
      pred_i = pred_i + 1;
    }
  }
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
