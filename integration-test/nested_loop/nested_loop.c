// clang-format off
#include "nested_loop.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

void nested_loop(in_int_t a[N], in_int_t b[N], inout_int_t c[N], inout_int_t d[N]) {
  for (int j = 0; j < 2; j++) {
    int i = 0;
    bool loopAgain1;
    // for -> for
    do { 
      c[i + j * 100] = a[i] * b[i];
      i++;
      loopAgain1 = i < 10;
    } while (loopAgain1);
  
    // for -> for -> if-else
    /* do { 
      int val_a = a[i];
      int val_b = b[i];
      int result;

      if (val_a > val_b) {
        result = val_a * val_b;
      } else {
        result = val_a - val_b + 42;
        d[i] = result;
      }

      // c[i + j * 100] = result;
      c[i * 100] = result;

      i++;
      loopAgain1 = i < 10;
    } while (loopAgain1);  */
  }
}


int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  inout_int_t c[N];
  inout_int_t d[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 5;
    b[j] = j;
    c[j] = 0;
    d[j] = 1;
  }

  CALL_KERNEL(nested_loop, a, b, c, d);
  return 0;
}