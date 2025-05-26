#include "float_basic.h"

#include "dynamatic/Integration.h"
#include <stdlib.h>


float __tester(float input_a, float input_b, float output_c, float output_d, 
                int output_e, int parameter_BITWIDTH, int parameter_bitrate, int parameter_consttwice);
int __init2();
float __init();

void float_basic(in_float_t A[30][30], in_float_t B[30][30], out_float_t y[30],
                 inout_float_t x[30]) {
  int i, j;
  int bitw = 31;
  int bitrate = 10;
  int consttwice = 10;

  for (i = 0; i < 30; i++) {
    float t_y = 0;

    float a_val = A[i][0];
    float b_val = A[i][1];
    float c_val = __init();
    float d_val = __init();
    int e_val = __init2();

    float result2 = c_val * d_val;
    
    float callres = __tester(a_val, b_val, c_val, d_val, e_val, bitw, bitrate, consttwice);
    float result = a_val + b_val + c_val - d_val - e_val;
    
    //float result2 = c_val * d_val;
    float result3 = c_val + c_val;

    t_y += result;
    t_y += result2;
    t_y += callres;
    t_y += result3;

    for (j = 0; j < 30; j++) {
      float t_x = x[j];
      if (A[i][j] <= B[i][j])
        t_y += A[i][j] / t_x;
      else
        t_y -= B[i][j] * t_x;
    }
    y[i] = t_y;
  }
}

int main(void) {
  in_float_t A[30][30];
  in_float_t B[30][30];
  out_float_t y[30];
  inout_float_t x[30];

  for (int j = 0; j < 30; ++j) {
    x[j] = (float)(rand() / RAND_MAX);
    for (int k = 0; k < 30; ++k) {
      A[j][k] = (float)(rand() / RAND_MAX * 10);
      B[j][k] = (float)(rand() / RAND_MAX * 10);
    }
  }

  CALL_KERNEL(float_basic, A, B, y, x);
  return 0;
}
