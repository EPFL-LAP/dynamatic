#include "cordic.h"
#include "dynamatic/Integration.h"
#include <math.h>
#include <stdlib.h>

float cordic(in_float_t theta, out_float_t a[2], in_float_t cordic_phase[1000],
             in_float_t current_cos, in_float_t current_sin) {

  float factor = 1.0;

  for (int i = 0; i < 1000; i++) {
    float sigma = (theta < 0) ? -1.0 : 1.0;
    float tmp_cos = current_cos;

    current_cos = current_cos - current_sin * sigma * factor;
    current_sin = tmp_cos * sigma * factor + current_sin;

    theta = theta - sigma * cordic_phase[i];
    factor = factor / 2;
  }
  a[0] = current_cos;
  a[1] = current_sin;
  return theta;
}

int main(void) {
  in_float_t thetas;
  out_float_t results[2];
  in_float_t cordic_phases[1000];
  in_float_t initial_cos = 1;
  in_float_t initial_sin = 0;
  // generate cordic_phases
  for (int i = 0; i < 1000; ++i) {
    double div = sqrtf(1 + powf(2, -2 * i));
    cordic_phases[i] = 1 / div;
  }

  thetas = 1;
  results[0] = 0;
  results[1] = 0;

  CALL_KERNEL(cordic, thetas, results, cordic_phases, initial_cos, initial_sin);
  return 0;
}