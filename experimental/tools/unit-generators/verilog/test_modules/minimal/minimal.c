#include "minimal.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>


int minimal(in_int_t x) {
  return x;
}

int main(void) {
  int x = rand();
  CALL_KERNEL(minimal, x);
  return 0;
}
