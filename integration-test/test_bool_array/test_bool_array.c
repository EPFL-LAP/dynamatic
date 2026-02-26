#include "dynamatic/Integration.h"
#include <stdbool.h>

_Bool test_bool_array(_Bool a[1]) { return a[0]; }

int main() {
  _Bool a[1] = {true};
  CALL_KERNEL(test_bool_array, a);
}
