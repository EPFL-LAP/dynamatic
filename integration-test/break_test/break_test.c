#include "break_test.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

void break_test(in_int_t a[N], in_int_t b[N], inout_int_t c[N],
                inout_int_t d[N]) {
  // I looked at cf_dyn_transformed
  // tail break for
  // int i;
  // for (i = 0; i < N; i++) {
  //   int temp = a[i] + b[i];
  //   c[i] = temp;
  //   if (a[i] > 3) {
  //     c[i + 1] = temp + 1;
  //     break;
  //   }
  // }
  // nontail break for
  // int i;
  // for (i = 0; i < N; i++) {
  //   int temp = a[i] + b[i];
  //   c[i] = temp;
  //   if (a[i] > 3) {
  //     c[i + 1] = temp + 1;
  //     break;
  //   }
  //   d[i] = temp;
  // }
  // tail break while
  // int i = 0;
  // while (b[i] < 5) {
  //   int temp = a[i] + b[i];
  //   c[i] = temp;
  //   i++;
  //   if (a[i] > 3) {
  //     c[i + 1] = temp + 1;
  //     break;
  //   }
  // }
  // nontail break while
  // int i = 0;
  // while (b[i] < 5) {
  //   int temp = a[i] + b[i];
  //   c[i] = temp;
  //   i++;
  //   if (a[i] > 3) {
  //     // c[i + 1] = temp + 1;
  //     break;
  //   }
  //   // d[i] = temp;
  // }
  // nontail break dowhile
  // int i = 0;
  // do {
  //   int temp = a[i] + b[i];
  //   c[i] = temp;
  //   d[i] = temp;
  //   i++;
  //   if (a[i] > 3) {
  //     c[i + 1] = temp + 1;
  //     break;
  //   }
  //   d[i] = temp;
  // } while (b[i] < 5);
  int i = 0;
  do {
    int temp = a[i] + b[i];
    c[i] = temp;
    i++;
    if (a[i] > 3) {
      c[i + 1] = temp + 1;
      break;
    }
    d[i] = temp;
  } while (b[i] < 5);
  // do {
  //   int temp = a[i] + b[i];
  //   c[i] = temp;
  //   i++;
  // } while (i < N);
  // d[0] = i;
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  inout_int_t c[N];
  inout_int_t d[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = rand() % 10;
    b[j] = rand() % 10;
  }

  CALL_KERNEL(break_test, a, b, c, d);

  return 0;
}
