#include "complexdiv.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void complexdiv(in_int_t a_i[1000], in_int_t a_r[1000], in_int_t b_i[1000],
                in_int_t b_r[1000], out_int_t c_i[1000], out_int_t c_r[1000]) {
  int i;

  for (i = 0; i < 1000; i++) {
    int bi = b_i[i];
    int br = b_r[i];
    int ai = a_i[i];
    int ar = a_r[i];
    int cr, ci;
    if (abs(br) >= abs(bi)) {
      int r = bi / br;
      int den = br + r * bi;
      cr = (ar + r * ai) / den;
      ci = (ai - r * ar) / den;
    } else {
      int r = br / bi;
      int den = bi + r * br;
      cr = (ar * r + ai) / den;
      ci = (ai * r - ar) / den;
    }
    c_r[i] = cr;
    c_i[i] = ci;
  }
}

int main(void) {
  in_int_t a_i[1000];
  in_int_t a_r[1000];
  in_int_t b_i[1000];
  in_int_t b_r[1000];
  out_int_t c_i[1000];
  out_int_t c_r[1000];

  for (int j = 0; j < 1000; ++j) {
    a_i[j] = 1;
    a_r[j] = 1;
    b_i[j] = 1;
    b_r[j] = 1;
  }

  CALL_KERNEL(complexdiv, a_i, a_r, b_i, b_r, c_i, c_r);
}
