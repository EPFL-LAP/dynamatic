#include <stdlib.h>
#define N 32
#define FACTOR 4
#include "dynamatic/Integration.h"

void test_mean(const int index[N], int result[1]) {

#pragma DYN array_partition array = values dimension = 1 style =               \
    cyclic factor = FACTOR
  int values[N];

#pragma clang loop unroll_count(FACTOR)
  for (int i = 0; i < N; i++)
    values[i] = i;

  int tmp = 0;
#pragma clang loop unroll_count(FACTOR)
  for (int i = 0; i < N; i++)
    tmp += values[index[i]];

  result[0] = tmp / N;
}

int main(void) {
  int index[N];
  int result[1] = {0};
  for (int i = 0; i < N; ++i) {
    index[i] = rand() % N;
  }
  CALL_KERNEL(test_mean, index, result);
  return 0;
}
