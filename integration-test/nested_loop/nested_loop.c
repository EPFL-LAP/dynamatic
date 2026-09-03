// clang-format off
#include "nested_loop.h"
#include "dynamatic/Integration.h"
#include "stdbool.h"
#include "stdlib.h"

void nested_loop(in_int_t a[N], in_int_t b[N], inout_int_t c[N], inout_int_t d[N]) {
  for (int j = 0; j < 2; j++) {
    int i = 0;
    int bound = 1000;
    int sum = 0;
    /* bool loopAgain;
    do {
      sum = a[i] * b[i];
      c[i + j * 400] = sum;
      i++;
      loopAgain = sum < bound;
      #pragma DYN speculate variable=loopAgain max_predictions=6 style=standard
    } while (loopAgain); */
    bool loopAgain1; 

    // Outer Subloop
    /* do { // mux13(?)
      c[i + j * 100] = a[i] * b[i];

      int k = 0;
      bool loopAgain2;

      // Inner Subloop 
      do {
        d[k + i + j * 100] = a[k] + b[k]; // mux14?
        k++;
        loopAgain2 = k < 5;
      } while (loopAgain2);

      i++;
      loopAgain1 = i < 10;
    } while (loopAgain1); */
     /* do { // mux13(?)
      c[i] = a[i];

      int k = 0;
      bool loopAgain2;

      // Inner Subloop 
      do {
        d[k + i + j * 100] = a[k]; // mux2, mux13, mux4, mux7
        k++;
        loopAgain2 = k < 5;
      } while (loopAgain2);

      i++;
      loopAgain1 = i < 10;
    } while (loopAgain1); */
    do { 
      int val_a = a[i];
      int val_b = b[i];
      int result;

      if (val_a > val_b) {
        result = val_a * val_b;
      } /* else if (val_a == val_b) { // mux5
        result = val_a + val_b;
      }  */ else {
        result = val_a - val_b + 42;
        d[i] = result; // mux8
      }

      // c[i + j * 100] = result;
      c[i * 100] = result;

      i++;
      loopAgain1 = i < 10;
    } while (loopAgain1); 
  }
}


int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  inout_int_t c[N];
  inout_int_t d[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 5;
    b[j] = j;
    c[j] = 0;
    d[j] = 1;
  }

  CALL_KERNEL(nested_loop, a, b, c, d);
  return 0;
}
