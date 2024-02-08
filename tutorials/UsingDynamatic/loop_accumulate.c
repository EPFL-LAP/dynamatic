#include "loop_accumulate.h"
#include "dynamatic/Integration.h"

unsigned loop_accumulate() {
  unsigned x = 2;
  for (unsigned i = 0; i < N; ++i) {
    if ((i & 1) == 0)
      x = x * x;
  }
  return x;
}

int main(void) {
  CALL_KERNEL(loop_accumulate);
  return 0;
}
