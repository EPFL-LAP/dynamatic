#include "dynamatic/Integration.h"
#include <stdint.h>

double test_double() { return 0.222837; }

int main() { CALL_KERNEL(test_double); }
