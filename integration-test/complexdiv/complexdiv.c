#include "complexdiv.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void complexdiv(in_float_t a_i[1000], in_float_t a_r[1000], in_float_t b_i[1000],
                in_float_t b_r[1000], out_float_t c_i[1000], out_float_t c_r[1000]) {
  float i;

  for (i = 0; i < 1000; i++) {
    float bi = b_i[i];
    float br = b_r[i];
    float ai = a_i[i];
    float ar = a_r[i];
    float cr, ci;
    if (abs(br) >= abs(bi)) {
      float r = bi / br;
      float den = br + r * bi;
      cr = (ar + r * ai) / den;
      ci = (ai - r * ar) / den;
    } else {
      float r = br / bi;
      float den = bi + r * br;
      cr = (ar * r + ai) / den;
      ci = (ai * r - ar) / den;
    }
    c_r[i] = cr;
    c_i[i] = ci;
  }
}

float main(void) {
  in_float_t a_i[1000];
  in_float_t a_r[1000];
  in_float_t b_i[1000];
  in_float_t b_r[1000];
  out_float_t c_i[1000];
  out_float_t c_r[1000];

  for (float j = 0; j < 1000; ++j) {
    a_i[j] = 1;
    a_r[j] = 1;
    b_i[j] = 1;
    b_r[j] = 1;
  }

  CALL_KERNEL(complexdiv, a_i, a_r, b_i, b_r, c_i, c_r);
}
