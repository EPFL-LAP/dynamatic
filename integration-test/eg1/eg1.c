#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 1000
void eg1(int a[N], int b[N], int c[N]) {
  int i = 0;
  int bound = 1000;
  int d = 0;
  while (d < bound) {
    d = a[i] * b[i];
    if (d < 5)
      break;
    c[i] = d;
    i++;
  }
}

int main(void) {
  int a[N];
  int b[N];
  int c[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 2;
    b[j] = j;
    c[j] = 0;
  }

  CALL_KERNEL(eg1, a, b, c);
  return 0;
}
