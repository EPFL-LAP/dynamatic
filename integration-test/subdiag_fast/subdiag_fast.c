// clang-format off
#include "subdiag_fast.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

int subdiag_fast(in_float_t d1[N], in_float_t d2[N], in_float_t e[N]) {
  int i = 0;
  bool cond_break = false;
  bool loop_again = false;
  do {
    float dd = d1[i] + d2[i + 1];
    float x = 0.001;
    i++;
    cond_break = (e[i]) <= x * dd;
    loop_again = !cond_break && i < N_DEC;
    #pragma DYN speculate variable=loop_again max_predictions=16 style=standard
  } while (loop_again);
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
