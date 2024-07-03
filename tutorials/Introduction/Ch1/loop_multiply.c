#include "loop_multiply.h"
#include "dynamatic/Integration.h"


unsigned loop_multiply(in_int_t a[N]) {
  unsigned x = 2;
  for (unsigned i = 0; i < N; ++i) {
    if (a[i] == 0)
      x = x * x;
  }
  return x;
}

/*
unsigned loop(in_int_t A[N],in_int_t B[N],in_int_t C[N], in_int_t addr[N]){
  unsigned sum = 0;
  for (unsigned i = 0; i < N; ++i) {
    sum = sum+A[addr[i]];
  }
  for (unsigned j = 0; j < N; ++j){
    for (unsigned k = 0; k < N; ++k){
       B[k] = A[addr[k]];
    }
    A[j] = C[j]*2;
  }
  return sum;
}*/


int main(void) {
  in_int_t a[N];
  // Initialize a to [0, 1, 0, 1, ...]
  for (unsigned i = 0; i < N; ++i)
    a[i] = i % 2;
  CALL_KERNEL(loop_multiply, a);
  return 0;
}

/*
int main(void){
  in_int_t A[N];
  in_int_t B[N];
  in_int_t C[N];
  in_int_t addr[N];
  for (unsigned i = 0; i < N; ++i){
    A[i] = i % 2;
    B[i] = i % 2;
    C[i] = i % 2;
    addr[i] = i % 2;
  }
  CALL_KERNEL(loop, A,B,C,addr);
  return 0;
}*/

