//===- share_test_2.c - Many sharable integer multiplications -----*- C -*-===//
//
// Many multiplication with large constants that cannot be strength-reduced.
//
//===----------------------------------------------------------------------===//

#include "share_test_2.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

void share_test_2(inout_int_t a[N]) {
  int acc = 0;
  for (unsigned i = 0; i < N; ++i) {
    int x = a[i];
    acc = acc + i + 123456;
    a[i] = acc * i * 456784 + 123456 * x * i + 444444 * x * acc;
  }
}

int main(void) {
  in_int_t a[N];
  for (unsigned j = 0; j < N; ++j)
    a[j] = j;

  CALL_KERNEL(share_test_2, a);
  return 0;
}
