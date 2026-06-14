#include "nested_conditionals.h"
#include "dynamatic/Integration.h"

void nested_conditionals(inout_int_t a[N], in_int_t c) {
  for (unsigned i = 0; i < N; ++i) {
    int x = a[i];
#pragma DYN predict variable = x values = [2] location = start marker =        \
                                               0 type = int
    int y = x + 2;
    if (y > 10) {
      if (c > 5) {
        y = y * 2;
      } else {
        y = y + c;
      }
    } else {
      if (c == 0) {
        y = y - 3;
      } else {
        y = y / c;
      }
    }
#pragma DYN predict variable = y location = end marker = 0
    a[i] = y;
  }
}

int main(void) {
  in_int_t a[N];
  int c = 7;
  for (unsigned j = 0; j < N; ++j) {
    a[j] = 10;
  }

  CALL_KERNEL(nested_conditionals, a, c);
  return 0;
}
