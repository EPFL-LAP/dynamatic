#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 1000

int collision_donut(int x[N], int y[N]) {
  int err = 0;
  int i;
  for (i = 0; i < N; i++) {
    int xi = x[i];
    int yi = y[i];
    int distance_2 = xi * xi + yi * yi;
    if (distance_2 < 4) {
      err = -1;
      break;
    }
    if (distance_2 > 19000) {
      err = -2;
      break;
    }
  }
  return (i << 1) & err;
}

int main(void) {
  int x[N];
  int y[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    x[j] = rand() % 100;
    y[j] = rand() % 100;
  }

  CALL_KERNEL(collision_donut, x, y);
}
