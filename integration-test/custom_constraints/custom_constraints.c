#include "custom_constraints.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int custom_constraints(in_int_t di[N]) {
  int tmp = 0;
  for (int i = 0; i < N; i++) {
    int a = i * i;
    int b = a * i;
    int c = b * b;
    int d = c * c;
    int e = d * d;
    int f = e * e;

    di[i] = f + 1;
  }
    
  return tmp;
}

int main(void) {
  in_int_t di[N];

  srand(13);
  
  for (int j = 0; j < N; ++j) {
    di[j] = rand() % 100;
  }

  CALL_KERNEL(custom_constraints, di);
  
  return 0;
}
