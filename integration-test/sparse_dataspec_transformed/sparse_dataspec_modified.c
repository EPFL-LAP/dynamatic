#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 512

float sparse_dataspec_transformed(float a[N], float x[N]) {
  float sum = 0.0f;
  float mul = 0.0f;
  int i = 0;
  do {
    do {
      mul = a[i] * x[i];
      i++;
    } while (mul == 0 && i < N);
    sum += mul;
  } while (i < N);
  return sum;
}

int main(void) {
  float a[N];
  float x[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = (float)(rand() % 10 < 8 ? 0 : rand() % 10);
    x[j] = (float)(rand() % 10);
  }

  CALL_KERNEL(sparse_dataspec_transformed, a, x);
  return 0;
}
