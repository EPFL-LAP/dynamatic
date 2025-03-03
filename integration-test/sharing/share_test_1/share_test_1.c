#include "share_test_1.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

int share_test_1(in_int_t a[1000], in_int_t b[1000]) {
  int i;
  int tmp = 0;

For_Loop1:
  for (i = 0; i < 1000; i++) {
    tmp += a[i] * b[999 - i] * 5;
  }

For_Loop2:
  for (i = 0; i < 1000; i++) {
    tmp += a[999 - i] * b[i];
  }

  return tmp;
}

int main(void) {
  in_int_t a[1000];
  in_int_t b[1000];

  for (int i = 0; i < 1000; ++i) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  CALL_KERNEL(share_test_1, a, b);
}
