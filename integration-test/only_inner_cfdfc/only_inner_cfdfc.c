#include "only_inner_cfdfc.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>


void only_inner_cfdfc(out_int_t a[N][M], out_int_t c[N]) {
  for (unsigned i = 0; i < N; ++i) {
    int acc = 0;
    for (unsigned j = 0; j < M; ++j) {
      acc += a[i][j];
    }
    c[i] = acc * acc + acc;
  }
}

int main(void) {
  int a[N][M];
  int c[N];

  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < M; ++x) {
      a[y][x] = rand() % 100;
    }
  }

  CALL_KERNEL(only_inner_cfdfc, a, c);
  return 0;
}
