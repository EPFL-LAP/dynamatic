#include "predictionN.h"
#include "dynamatic/Integration.h"

void predictionN(inout_float_t a[N], inout_float_t b[N], in_float_t c) {
  for (unsigned i = 0; i < N; ++i) {
    float x = a[i];
    float y;
    float d = ((x + 1.0f) * 10.0f);
    if (d == 100.0f) {
      y = 5.0f;
    } else {
      if (d == 200.0f) {
        y = 6.0f;
      } else {
        y = x;
      }
    }
    if (y == 5.0f) {
      b[i] = (5.0f + c) * 10.0f;
    } else if (y == 6.0f) {
      b[i] = (6.0f + c) * 10.0f;
    } else if (y == 7.0f) {
      b[i] = (7.0f + c) * 10.0f;
    } else {
      b[i] = (y + c) * 10.0f;
    }
    // b[i] = (y + c) * 10.0f;
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

  CALL_KERNEL(predictionN, a, b, c);
  return 0;
}
