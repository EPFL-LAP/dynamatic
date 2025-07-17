#include "comparator.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int comparator(in_int_t a, in_int_t b){
  int min = a, max = b, result = 0;
  if (a> b)
    result = a;
  else if (b> a)
    result = b;
  return result;

}


int main(void) {
  in_int_t A[10];
  in_int_t n,a,b;
  
  srand(13);
  //n = rand() % 100;
  n=15;
  a=15;b=14;
  for (unsigned j = 5; j < 15; ++j)
      A[j-5] =  j;
  //CALL_KERNEL(comparator, A[2], A[3]);
  CALL_KERNEL(comparator, a,b);
  return 0;
}