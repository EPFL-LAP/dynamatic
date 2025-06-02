#include "dynamatic/Integration.h"

#define A 4
#define B 8
#define C 16

void test_flatten_array(int input[A][B][C], int output[A][B][C]) {
  for (int a = 0; a < A; a++) {
    for (int b = 0; b < B; b++) {
      for (int c = 0; c < C; c++) {
        output[a][b][c] = input[a][B - b - 1][C - c - 1];
      }
    }
  }
}

int main() {
  int input[A][B][C];
  int output[A][B][C];

  // fill input with random integers
  for (int a = 0; a < A; a++) {
    for (int b = 0; b < B; b++) {
      for (int c = 0; c < C; c++) {
        input[a][b][c] = a * B + b * B + c;
      }
    }
  }

  CALL_KERNEL(test_flatten_array, input, output);
  return 0;
}
