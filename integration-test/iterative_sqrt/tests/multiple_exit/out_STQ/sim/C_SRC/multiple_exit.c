#include "multiple_exit.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int multiple_exit(in_int_t arr[N]) {
    for (int i = 0; i < N; i++) {
        if (arr[i] == -1) {
           break ; 
        }
        if (arr[i] == 0) {
            return 1;
        }
        
        arr[i] += 1;
    }
    return 2;
}

int main(void) {
  in_int_t arr[N];
  in_int_t n;
  
  srand(13);
  //n = rand() % 100;
  n=10;
  for (int j = 0; j < N; j++)
      arr[j] =  j-8;
  CALL_KERNEL(multiple_exit, arr);
  return 0;
}