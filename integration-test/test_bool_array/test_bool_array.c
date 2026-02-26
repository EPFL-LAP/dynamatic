#include "dynamatic/Integration.h"
#include <stdbool.h>

_Bool test_bool_array(_Bool a[2]) { return a[0] || a[1]; }

int main() {
  _Bool a[2] = {true, false};
  CALL_KERNEL(test_bool_array, a);
}
