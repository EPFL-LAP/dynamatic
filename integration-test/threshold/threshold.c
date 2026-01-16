//===- threshold.c - Threshold on RGB channels --------------------*- C -*-===//
//
// Implements the threshold kernel.
//
//===----------------------------------------------------------------------===//

#include "threshold.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void threshold(in_int_t th, inout_int_t red[N], inout_int_t green[N],
               inout_int_t blue[N]) {
  //for (unsigned i = 0; i < N; i++) {
  int i = red[2]+ th;
  int j = red[3];
  int x = 7;
    int sum = red[i] + green[i] + blue[i];
    if (sum <= th) {
      red[i] = 0;
      green[i] = 0;
      blue[i] = j;
      x +=5;
    }
    else{
      red[i] = 5;
      green[i] = 7;
      blue[i] = red[2];
      x +=25;
    }
    red[0] =  x+1;
  //}
}

int main(void) {
  in_int_t th;
  inout_int_t red[N];
  inout_int_t green[N];
  inout_int_t blue[N];

  th = rand() % 100;
  for (int j = 0; j < N; ++j) {
    red[j] = (rand() % 100);
    green[j] = (rand() % 100);
    blue[j] = (rand() % 100);
  }

  CALL_KERNEL(threshold, th, red, green, blue);
  return 0;
}
