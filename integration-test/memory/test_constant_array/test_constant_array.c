#include "dynamatic/Integration.h"
#include "stdlib.h"
#define N 3

const int constant_array[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

int test_constant_array(int di[N][N]) {
  int tmp = 0;
  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      tmp += di[i][j] + constant_array[i][j];
    }
  }
  return tmp;
}

int main(void) {
  int di[N][N];

  srand(13);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      di[i][j] = rand() % 100;
    }
  }

  CALL_KERNEL(test_constant_array, di);
  return 0;
}
