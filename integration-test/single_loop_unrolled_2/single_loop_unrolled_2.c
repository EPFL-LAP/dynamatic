#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 50

void single_loop_unrolled_2(int a[N], int b[N], int c0[N], int c1[N]) {
  int i = 0;
  int bound = 50;
  int sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c0[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c1[i] = sum;
    i++;
  }
}

int main(void) {
  int a[N];
  int b[N];
  int c0[N];
  int c1[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 2;
    b[j] = j;
    c0[j] = 0;
    c1[j] = 0;
  }

  CALL_KERNEL(single_loop_unrolled_2, a, b, c0, c1);
  return 0;
}
