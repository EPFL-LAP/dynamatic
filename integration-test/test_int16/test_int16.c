#include "dynamatic/Integration.h"
#include <stdint.h>

int16_t test_int16() {
  return -1;
}

int main() {
  CALL_KERNEL(test_int16);
}
