#include "dynamatic/Integration.h"
#include <stdint.h>

double test_fneg(double a) { return -a; }

int main() {
  double a = 5;
  CALL_KERNEL(test_fneg, a);
}
