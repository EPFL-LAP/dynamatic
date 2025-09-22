#include "subdiag.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int subdiag(in_float_t d[N], in_float_t e[N]) {
  int i;

  for (i = 0; i < N_DEC; i++) {
    float dd = d[i] + d[i + 1];
    float x = 0.001f;
    if ((e[i]) <= x * dd)
      break;
  }

  return i;
}

int main(void) {
  in_float_t d[N];
  in_float_t e[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    d[j] = j;
    e[j] = (300.0f - j) * 0.001f;
  }

  CALL_KERNEL(subdiag, d, e);
  return 0;
}
