/**#include "iterative_sqrt.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"


int iterative_sqrt(in_int_t A[10]) {
  int x= 5;
   return A[1]+x;
}


int main(void) {
  in_int_t A[10];
  //in_int_t n;

  //srand(13);
  //n = rand() % 100;
  //n = 25;
  //for (unsigned j = 0; j < 10; ++j)
    //  A[j] =  j;

  CALL_KERNEL(iterative_sqrt, A);
  return 0;
}*/



#include "iterative_sqrt.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int iterative_sqrt(in_int_t n) {
  int low = 0, high = n, mid;
  while (low <= high) {
    // Divide by 2
    mid = ((low + high) >> 1);
    if (mid * mid == n) {
      return mid;
    }
    if (mid * mid < n) {
      low = mid + 1;
    }
    else {
      high = mid - 1;
    }
  }
  return high;
}

int main(void) {
  in_int_t n;

  srand(13);
  n = rand() % 100;

  CALL_KERNEL(iterative_sqrt, n);
  return 0;
}