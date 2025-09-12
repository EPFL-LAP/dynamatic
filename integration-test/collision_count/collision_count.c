#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 1000

int collision_count(int x[N], int y[N]) {
  int count1 = 0;
  int count2 = 0;
  for (int i = 0; i < N; i++) {
    int xi = x[i];
    int yi = y[i];
    int distance_2 = xi * xi + yi * yi;
    if (distance_2 > 19000) {
      break;
    }
    if (xi > 0 && yi > 0) {
      count1++;
    } else if (xi < 0 && yi < 0) {
      count2++;
    }
  }
  return count1 + count2;
}

int main(void) {
  int x[N];
  int y[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    x[j] = rand() % 200 - 100;
    y[j] = rand() % 200 - 100;
  }

  CALL_KERNEL(collision_count, x, y);
}
