#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 1000
int linear_search(int a[N], int item) {
  for (int i = 0; i < N; i++) {
    int data = a[i];
    // condition
    if (data * data == item) {
      // Item found
      return i;
    }
  }
  return -1;
}

int main(void) {
  int a[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = rand() % 100;
  }

  int item = 42;

  CALL_KERNEL(linear_search, a, item);
  return 0;
}
