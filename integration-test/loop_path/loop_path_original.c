#include "loop_path.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

void loop_path(in_int_t a[N], in_int_t b[N], inout_int_t c[N]) {
  int i;

  for (i = 0; i < N; i++) {
    int temp = a[i] + b[i];
    int x = 5;
    c[i] = temp;
    if ((1000 - temp) <= x * temp) {
      break;
    }
  }
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  inout_int_t c[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = rand() % 10;
    b[j] = rand() % 10;
  }

  CALL_KERNEL(loop_path, a, b, c);
  return 0;
}
