#include "divergent_paths.h"
#include "dynamatic/Integration.h"

void divergent_paths(inout_int_t a[N], inout_int_t b[N]) {
  for (unsigned i = 0; i < N; ++i) {
    int x = a[i];
#pragma DYN predict variable = x values = [7] location = start marker =        \
                                               3 type = int
    int split_val = x ^ 0xFF;

    int path_a = split_val + 2;
#pragma DYN predict variable = path_a location = end marker = 3

    int path_b = split_val * 3;
#pragma DYN predict variable = path_b location = end marker = 3

    a[i] = path_a;
    b[i] = path_b;
  }
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  for (unsigned j = 0; j < N; ++j) {
    a[j] = 10;
    b[j] = 0;
  }
  a[10] = 20;
  a[20] = 15;
  a[30] = 20;

  CALL_KERNEL(divergent_paths, a, b);
  return 0;
}
