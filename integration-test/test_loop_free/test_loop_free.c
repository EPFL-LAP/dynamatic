#include "dynamatic/Integration.h"

int test_loop_free(int a, int b, int c, int d) {

  if (a > 0) {
    return b + c + d;
  } else if (b < 0) {
    return a + d;
  } else {
    return a;
  }
  return 0;
}

int main() {
  int a = 0;
  int b = 1;
  int c = 2;
  int d = 3;

  CALL_KERNEL(test_loop_free, a, b, c, d);
  return 0;
}
