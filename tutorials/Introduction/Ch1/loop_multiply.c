#include "loop_multiply.h"
#include "dynamatic/Integration.h"

void loop_multiply(in_int_t alpha, inout_int_t tmp[NI][NJ]) {
  // unsigned x = 2;
  // for (unsigned i = 0; i < N; ++i) {
  //   if (a[i] == 0)
  //     x = x * x;
  // }
  // return x;

  /*
  for (unsigned j = 0; j < NJ; j++) {

    tmp[j][j] = tmp[j][j] + alpha + j;

    if (alpha > 5) {
      // if (alpha > 7)
      //     int sum = tmp[j][j];
      //     if (sum > 5)
      //        for (unsigned k = 0; k < NK; ++k)
      tmp[j][j] = tmp[j][j] + alpha;
    } else {
      tmp[j][j] = tmp[j][j] + alpha + j;
    }
  }*/
  //} // else {
  // tmp[j][j] = tmp[j][j] + alpha;
  //}*/

  for (int j = 0; j < NJ; j++) {
    tmp[j][j] = tmp[j][j] + alpha + j;

    if (alpha > 5) {
      if (alpha > 7)
        tmp[j][j] = tmp[j][j] + alpha;
    } else {
      tmp[j][j] = tmp[j][j] + alpha + j;
    }
  }
}

int main(void) {
  in_int_t alpha;
  in_int_t beta;
  inout_int_t tmp[NI][NJ];
  in_int_t A[NI][NK];
  in_int_t B[NK][NJ];
  in_int_t C[NK][NL];
  inout_int_t D[NI][NL];

  alpha = 6;
  beta = 100;
  for (unsigned i = 0; i < NI; ++i) {
    for (unsigned k = 0; k < NK; ++k)
      A[i][k] = 100 % 100;
    for (unsigned l = 0; l < NL; ++l)
      D[i][l] = 100 % 100;
  }

  for (unsigned k = 0; k < NK; ++k) {
    for (unsigned j = 0; j < NJ; ++j)
      B[k][j] = 100 % 100;
    for (unsigned l = 0; l < NL; ++l)
      C[k][l] = 100 % 100;
  }

  CALL_KERNEL(loop_multiply, alpha, tmp);
  // CALL_KERNEL(kernel_2mm, alpha, beta, tmp, A, B, C, D);

  return 0;
}