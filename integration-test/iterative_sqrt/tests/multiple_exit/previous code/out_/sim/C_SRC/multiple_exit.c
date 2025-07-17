#include "multiple_exit.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

void multiple_exit(in_int_t arr[10], in_int_t size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == -1) {
            break;  // Exit path 1: break out of the loop early
        }
        if (arr[i] == 0) {
            return; // Exit path 2: exit the function completely
        }
        // Normal loop body
        arr[i] += 1;
    }
}

int main(void) {
  in_int_t A[10];
  in_int_t n;
  
  srand(13);
  //n = rand() % 100;
  n=10;
  for (unsigned j = 0; j < n; ++j)
      A[j] =  j-10;
  CALL_KERNEL(multiple_exit, A, n);
  return 0;
}