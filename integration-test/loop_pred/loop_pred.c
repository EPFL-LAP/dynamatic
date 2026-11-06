#include "loop_pred.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

float loop_pred(in_float_t a[N], in_float_t x[N]) {
  float sum = 0.0f;
  int i = 0;
  float mul;
  for (i = 0; i < N; i++) {
    mul = 0.5f * x[i] + 3.0f + (a[i] * sum);
#pragma DYN predict variable = mul values = [5] location = start marker =      \
                                                 0 type = float
    sum = (sum * mul) + 12.0f;
#pragma DYN predict variable = sum location = end marker = 0
  }
  return sum;
}

int main(void) {
  in_float_t a[N];
  in_float_t x[N];

  for (int j = 0; j < N; ++j) {
    a[j] = 0.0f;
    if (j < 0) {
      x[j] = 7.0f;
    } else {
      x[j] = 4.0f;
    }
  }

  CALL_KERNEL(loop_pred, a, x);
  return 0;
}
