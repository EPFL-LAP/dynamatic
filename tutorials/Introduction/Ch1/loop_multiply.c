#include "loop_multiply.h"
#include "dynamatic/Integration.h"

unsigned loop_multiply(in_int_t a[N]) {
  // unsigned x = 2;
  // // original example of the tutorial
  // for (unsigned i = 0; i < N; ++i) {
  //   if (a[i] == 0)
  //     x = x * x;
  // }

  // slightly complicated if-then-else example
  // int i = 0;
  // if( x < a[i]) {
  //   x = x + a[i+1];
  //   if(x < a[7])
  //     x = x + a[2] + a[7];
  // }
  // else
  //   x = x + a[i+1] + a[i+2];

  unsigned x = 2;
  unsigned y = a[2];
  unsigned z = 6;
  int i = 0;
  if(x < a[i]) {
    y = a[i+1];
  }
  else {
    a[i] = a[i+1];
  }
  x = y + z;
  
  // loop with if-then-else example
  // for(unsigned k = 0; k < N; k++)
  //   for(unsigned i = 0; i < N-2; i++) {
  //     if( x < a[i])
  //       x = x + a[i+1];
  //     else
  //       x = x + a[i+1] + a[i+2];
  //   }
 
  return x;
}

int main(void) {
  in_int_t a[N];
  // Initialize a to [0, 1, 0, 1, ...]
  for (unsigned i = 0; i < N; ++i)
    a[i] = i % 2;
  CALL_KERNEL(loop_multiply, a);
  return 0;
}
