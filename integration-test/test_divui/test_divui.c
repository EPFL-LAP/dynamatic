// This benchmark tests both divui and remui are implemented correctly

#include "dynamatic/Integration.h"
#include <stdint.h>

uint32_t test_divui(uint32_t var1) { return var1 / 1083 + 1099 % var1; }

int main() {
  uint32_t var1 = 1084;
  CALL_KERNEL(test_divui, var1);
}
