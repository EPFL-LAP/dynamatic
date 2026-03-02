//===- test_memory_deps.c -----------------------------------------*- C -*-===//

#include "test_memory_deps.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void test_memory_deps(in_int_t load_addrs[1000], in_int_t store_addrs[1000], inout_int_t data[1000], in_int_t n) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += data[load_addrs[i]];
    data[store_addrs[i]] = i;
  }
  data[0] = sum;
}

int main(void) {
  in_int_t load_addrs[1000];
  in_int_t store_addrs[1000];
  inout_int_t data[1000];

  in_int_t n = 1000;
  for (int i = 0; i < n; ++i) {
    // addresses alternate randomly between 1 and 2, creating RAW and WAR hazards
    load_addrs[i] = (rand() % 4) + 1;
    load_addrs[i] = (rand() % 4) + 1;
    store_addrs[i] = (i == 0) ? 1 : 2;
    data[i] = i;
  }

  CALL_KERNEL(test_memory_deps, load_addrs, store_addrs, data, n);
  return 0;
}
