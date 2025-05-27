#include "iterative_division.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

// Maybe we can create a stdlib for dynamatic? Creating a select for this is a
// overkill...
#define ABS(x) (x >= 0) ? (x) : (-x)

int iterative_division(in_int_t dividend, in_int_t divisor) {
  if (divisor == 0)
    return -1;
  int quotient = 0;
  int sign = ((dividend < 0) ^ (divisor < 0)) ? -1 : 1;

  dividend = ABS(dividend);
  divisor = ABS(divisor);

  while (dividend >= divisor) {
    dividend -= divisor;
    quotient++;
  }
  return (sign == -1) ? -quotient : quotient;
}

int main(void) {
  in_int_t dividend;
  in_int_t divisor;

  srand(13);
  dividend = rand() % 100;
  divisor = rand() % 100;

  CALL_KERNEL(iterative_division, dividend, divisor);
  return 0;
}
