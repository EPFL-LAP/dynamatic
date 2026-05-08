#include "prediction.h"
#include "dynamatic/Integration.h"

void prediction(inout_float_t a[N], inout_float_t b[N], in_float_t c) {
  for (unsigned i = 0; i < N; ++i) {
    float x = a[i];
    float y;
    switch ((int)((x + 1.0) * 10.0)) {
    case 100:
      y = 5.0;
      break;
    default:
      y = x;
    }
    /*
    if (y == 5.0) {
      b[i] = (5.0 + c) * 10.0;
    } else {
      b[i] = (y + c) * 10.0;
    } */
    b[i] = (y + c) * 10.0;
  }
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  float c = 7.0;
  for (unsigned j = 0; j < N; ++j) {
    a[j] = 10;
    b[j] = 0;
  }
  a[10] = 20;
  a[20] = 15;
  a[30] = 20;

  CALL_KERNEL(prediction, a, b, c);
  return 0;
}
