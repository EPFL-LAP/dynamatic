#include "if_convert.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

void if_convert(in_int_t a[N], inout_int_t b[N]) {
  int i = 1;
  do {
    int prev_i = i;
    int tmp = a[i];
    if (i * tmp < 10000) {
      i++;
    }
    i++;
    b[i] = prev_i;
  } while (i < N2);
}

int main(void) {
  in_int_t a[N];
  inout_int_t b[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = rand() % 5000;
    b[j] = 0;
  }

  CALL_KERNEL(if_convert, a, b);
  return 0;
}
