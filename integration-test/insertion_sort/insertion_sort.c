#include "insertion_sort.h"
#include "../integration_utils.h"
#include <stdlib.h>

void insertion_sort(inout_int_t A[N], in_int_t n) {
  for (int i = 1; i < n; ++i) {
    int x = A[i];
    int j = i;
    while (j > 0 && A[j - 1] > x) {
      A[j] = A[j - 1];
      --j;
    }
    A[j] = x;
  }
}

int main(void) {
  inout_int_t a[N];
  for (int j = 0; j < N; ++j)
    a[j] = rand() % 10;

  CALL_KERNEL(insertion_sort, a, N);
  return 0;
}
