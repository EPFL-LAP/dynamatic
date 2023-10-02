//===- sobel.c - Sobel filter -------------------------------------*- C -*-===//
//
// Implements the sobel kernel.
//
//===----------------------------------------------------------------------===//

#include "sobel.h"

int sobel(int in[N], int gX[9], int gY[9], int out[N]) {
  int sum = 0;
  for (unsigned y = 0; y < 15; y++) {
    for (unsigned x = 0; x < 15; x++) {
      int sumX = 0;
      int sumY = 0;

      bool t1, t2, c1, c2;

      /* image boundaries */
      t1 = y == 0;
      t2 = y == 5;
      c1 = t1 || t2;
      c1 = !c1;

      t1 = x == 0;
      t2 = x == 5;
      c2 = t1 || t2;
      c2 = !c2;

      if (c2) {
        for (int i = -1; i <= 1; i++) {
          for (int j = -1; j <= 1; j++) {
            sumX += in[x] * gX[3 * i + j + 4];
            sumY += in[x] * gY[3 * i + j + 4];
          }
        }

        if (sumX > 255)
          sumX = 255;
        if (sumX < 0)
          sumX = 0;

        /*-------Y GRADIENT APPROXIMATION-------*/
        if (sumY > 255)
          sumY = 255;
        if (sumY < 0)
          sumY = 0;

        sum += sumX + sumY;
      }

      out[x + y] = 255 - (unsigned char)(sum);
    }
  }

  return sum;
}
