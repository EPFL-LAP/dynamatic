#include "path_profiling.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void path_profiling(out_int_t a[LOOP_BOUND], out_int_t b[LOOP_BOUND],
                    in_int_t var[LOOP_BOUND]) {
  for (int i = 0; i < LOOP_BOUND; ++i) {
    if (var[i] % 2 == 0) {
      a[i] = 2;
    } else {
      a[i] = 3;
    }
    if (var[i] % 3 == 0) {
      b[i] = 2;
    } else {
      b[i] = 3;
    }
  }
}

int main(void) {
  out_int_t a[LOOP_BOUND];
  out_int_t b[LOOP_BOUND];
  in_int_t var[LOOP_BOUND];

  for (int j = 0; j < LOOP_BOUND; ++j) {
    a[j] = 0;
    b[j] = 0;
    var[j] = rand() % 100;
  }

  CALL_KERNEL(path_profiling, a, b, var);
  return 0;
}
