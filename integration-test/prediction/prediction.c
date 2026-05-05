#include "prediction.h"
#include "dynamatic/Integration.h"

void loop(inout_int_t a[N], inout_int_t b[N]) {
  int x = 0;
  for (unsigned i = 1; i < N; ++i) {
    switch (a[i] * 10) {
    case 100:
      x = 5;
    case 200:
      x = 7;
    default:
      x = a[i];
    }
    if (x == 5) {
      b[i] = 25;
    } else if (x == 7) {
      b[i] = 49;
    } else {
      b[i] = x * x;
    }
  }
}

int main(void) {
  in_float_t a[N];
  in_int_t b[N];
  for (unsigned j = 0; j < N; ++j) {
    a[j] = 10;
    b[j] = 0;
  }
  a[10] = 20;
  a[20] = 15;
  a[30] = 20;

  CALL_KERNEL(loop, a, b);
  return 0;
}
