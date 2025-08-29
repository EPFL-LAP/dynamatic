#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"
#include "subdiag.h"

int subdiag(in_float_t d[N], in_float_t e[N]) {
  int i = 0;
  int next_i = 0;
  bool cond_break = false;
  // Require code transformation into a single BB loop
  do {
    i = next_i;
    float dd = d[i] + d[i + 1];
    float x = 0.001;
    next_i = i + 1;
    cond_break = (e[i]) <= x * dd;
  } while (next_i < N_DEC && !cond_break);
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
