#include "vector_rescale.h"
#include "../integration_utils.h"
#include <stdlib.h>

void vector_rescale(inout_int_t a[N], in_int_t c) {
  for (unsigned i = 0; i < N; ++i)
    a[i] = a[i] * c;
}

int main(void) {
  in_int_t a[N];
  in_int_t c;

  srand(13);
  c = rand() % 100;
  for (unsigned j = 0; j < N; ++j)
    a[j] = rand() % 100;

  CALL_KERNEL(vector_rescale, a, c);
  return 0;
}
