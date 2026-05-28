#include "prediction.h"
#include "dynamatic/Integration.h"

void prediction(inout_float_t a[N], inout_float_t b[N], in_float_t c) {
  for (unsigned i = 0; i < N; ++i) {
    float x = a[i];
    float y;
    switch ((int)((x + 1.0f) * 10.0f)) {
    case 100:
      y = 5.0f;
      break;
    default:
      y = x;
    }
    /*
    if (y == 5.0f) {
      b[i] = (5.0f + c) * 10.0f;
    } else {
      b[i] = (y + c) * 10.0f;
    } */
#pragma DYN predict variable = y values = [7.0, 7.5] location = start marker = 0
    b[i] = (y + c) * 10.0f;
#pragma DYN predict variable = y values = [7.0, 7.5] location = start marker = 1
    a[i] = (y - c) * 10.0f;
  }
}

int main(void) {
  in_float_t a[N];
  in_float_t b[N];
  float c = 7.0f;
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
