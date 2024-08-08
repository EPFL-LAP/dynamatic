#include "easy_sum.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int easy_sum(in_int_t a, in_int_t b, in_int_t cst, in_int_t c) {
  int tmp = 0;
  for (int i = 0; i < cst; i++)
    tmp += a * b;
  tmp += c;
  return tmp;
}

int main(void) {
  in_int_t a = 2;
  in_int_t b = 3;
  in_int_t cst = 4;
  in_int_t c = 5;

  CALL_KERNEL(easy_sum, a, b, cst, c);
  return 0;
}
