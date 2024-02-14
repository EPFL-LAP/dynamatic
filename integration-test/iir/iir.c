
#include "iir.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

int iir(in_int_t y[N], in_int_t x[N], in_int_t a, in_int_t b) {
  int tmp = y[0];
  for (unsigned i = 1; i < N; i++) {
    tmp = a * tmp + b * x[i];
    y[i] = tmp;
  }
  return tmp;
}

int main(void) {
  in_int_t y[N];
  in_int_t x[N];
  in_int_t b;
  in_int_t a;

  srand(13);
  a = rand();
  b = rand();
  for (int j = 0; j < N; ++j) {
    y[j] = rand() % N;
    x[j] = rand() % N;
  }

  CALL_KERNEL(iir, y, x, a, b);
  return 0;
}
