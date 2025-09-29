#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 1000
void nested_loop(int a[N], int b[N], int c[N]) {
  for (int j = 0; j < 20; j++) {
    int i = 0;
    int bound = 1000;
    int sum = 0;
    while (sum < bound) {
      sum = a[i] * b[i];
      c[i + j * 40] = sum;
      i++;
    }
  }
}

int main(void) {
  int a[N];
  int b[N];
  int c[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 20;
    b[j] = j;
    c[j] = 0;
  }

  CALL_KERNEL(nested_loop, a, b, c);
  return 0;
}
