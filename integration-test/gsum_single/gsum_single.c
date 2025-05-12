
//------------------------------------------------------------------------
// Jianyi Cheng, DSS
// https://zenodo.org/record/3561115
//------------------------------------------------------------------------

#include "gsum_single.h"
#include <stdlib.h>

int gsum_single(in_float_t a[N]) {
  int i;
  float d;
  int kk = 0;

  float s = 0.0;
  for (i = 0; i < 1000; i++) {
    d = a[i];
    if (d >= 0)
      s += (((((d + (float)0.64) * d + (float)0.7) * d + (float)0.21) * d +
             (float)0.33) *
            d);
  }

  return i;
}

#define AMOUNT_OF_TEST 1

int main(void) {
  in_float_t a[AMOUNT_OF_TEST][N];

  in_float_t b[AMOUNT_OF_TEST][N];

  out_float_t c[AMOUNT_OF_TEST][N];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i) {
    for (int j = 0; j < N; ++j) {
      a[i][j] = (float)1 - j;
      b[i][j] = (float)j + 10;

      c[i][j] = (float)j + 10;

      if (j % 100 == 0)
        a[i][j] = j;
    }
  }

  int i = 0;
  gsum_single(a[i]);
}
