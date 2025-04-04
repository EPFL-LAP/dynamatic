#include "factorial.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

#define UNROLL_FACTOR 3

int factorial(in_int_t n) {
  int result = 1;

  while (n >= UNROLL_FACTOR) {
#if UNROLL_FACTOR == 1
    result *= n;
#elif UNROLL_FACTOR == 2
    result *= n * (n - 1);
#elif UNROLL_FACTOR == 3
    result *= n * (n - 1) * (n - 2);
#elif UNROLL_FACTOR == 4
    result *= n * (n - 1) * (n - 2) * (n - 3);
#elif UNROLL_FACTOR == 5
    result *= n * (n - 1) * (n - 2) * (n - 3) * (n - 4);
#else
    #error "Unsupported UNROLL_FACTOR. Please define it."
#endif
    n -= UNROLL_FACTOR;
  }

  // Handle remaining numbers
  while (n > 0) {
      result *= n;
      n--;
  }

  return result;
}

int main(void) {
  in_int_t n;

  srand(13);
  n = rand() % 100;

  CALL_KERNEL(factorial, n);
  return 0;
}
