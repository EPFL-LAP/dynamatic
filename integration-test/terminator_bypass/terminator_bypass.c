#include "terminator_bypass.h"
#include "dynamatic/Integration.h"

void terminator_bypass(inout_int_t a[N], in_int_t c) {
  for (unsigned i = 0; i < N; ++i) {
    int x = a[i];
#pragma DYN predict variable = x values = [1] location = start marker =        \
                                               5 type = int
    int modified = x * 2;
    if (modified == c) {
      // introduces a conditional branch terminator inside the scope
      a[i] = 0;
      continue;
    }
#pragma DYN predict variable = modified location = end marker = 5
    a[i] = modified;
  }
}

int main(void) {
  in_int_t a[N];
  int c = 7;
  for (unsigned j = 0; j < N; ++j) {
    a[j] = 10;
  }

  CALL_KERNEL(terminator_bypass, a, c);
  return 0;
}
