// clang-format off
#include "subdiag.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

int subdiag(in_float_t d[N], in_float_t e[N]) {
  int i = 0;
  bool cond_break = false;
  bool loop_again = false;
  do {
    float dd = d[i] + d[i + 1];
    float x = 0.001;
    i++;
    cond_break = (e[i]) <= x * dd;
    loop_again =  i < N_DEC && !cond_break;
    #pragma DYN speculate variable = loop_again max_predictions = 8 style = standard
  } while (loop_again);
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
