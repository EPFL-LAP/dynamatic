#include "subdiag.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int subdiag(in_float_t d[N], in_float_t e[N]) {
  int i;

  for (i = 0; i < N_DEC; i++) {
    float dd = d[i] + d[i + 1];
    float x = 0.001;
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
    d[j] = rand() % 10;
    e[j] = rand() % 10;
  }

  CALL_KERNEL(subdiag, d, e);
  return 0;
}
