#include "subdiag_fast.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

int subdiag_fast(in_float_t d1[N], in_float_t d2[N], in_float_t e[N]) {
  int i = 0;
  for (i = 0; i < N_DEC; i++) {
    float dd = d1[i] + d2[i];
    float x = 0.001f;
    if ((e[i]) <= x * dd)
      break;
  }
  return i;
}

int main(void) {
  in_float_t d1[N];
  in_float_t d2[N];
  in_float_t e[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    d1[j] = j;
    d2[j] = j;
    e[j] = (300.0f - j) * 0.001f;
    // e[j] = (3.0f - j) * 0.001f;
  }

  CALL_KERNEL(subdiag_fast, d1, d2, e);
  return 0;
}
