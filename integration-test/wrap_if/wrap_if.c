#include "wrap_if.h"
#include "dynamatic/Integration.h"

void wrap_if(inout_int_t a[N], inout_int_t b[N], in_int_t c) {
  for (unsigned i = 0; i < N; ++i) {
    int x = a[i];
    int y = x * 3;
    if (x > 0) {
#pragma DYN predict variable = y values = [3, -4, 5] location = start marker = \
                                                      0 type = int
      b[i] = (y + c) * 10;
    }
    a[i] = (y - c) * 11;
  }
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  int c = 7;
  for (unsigned j = 0; j < N; ++j) {
    a[j] = 10;
    b[j] = 0;
  }
  a[10] = 20;
  a[20] = 15;
  a[30] = 20;

  CALL_KERNEL(wrap_if, a, b, c);
  return 0;
}
