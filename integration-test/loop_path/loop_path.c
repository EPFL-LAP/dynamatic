#include "loop_path.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

void loop_path(in_int_t a[N], in_int_t b[N], inout_int_t c[N]) {
  int i = 0;
  bool break_flag = false;

  do {
    int temp = a[i] + b[i];
    int x = 5;
    c[i] = temp;
    i++;
    break_flag = (1000 - temp) <= x * temp;
  } while (i < N && !break_flag);
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  inout_int_t c[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = j;
    b[j] = 1;
  }

  CALL_KERNEL(loop_path, a, b, c);
  return 0;
}
