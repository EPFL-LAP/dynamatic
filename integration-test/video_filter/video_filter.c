//===- video_filter.c - RGB color transformations -----------------*- C -*-===//
//
// Implements the video_filter kernel.
//
//===----------------------------------------------------------------------===//

#include "video_filter.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

void video_filter(in_int_t offset, in_int_t scale, inout_int_t pixelR[N][N],
                  inout_int_t pixelG[N][N], inout_int_t pixelB[N][N]) {
  for (unsigned row = 0; row < N; row++) {
    for (unsigned col = 0; col < N; col++) {
      pixelR[row][col] = ((pixelR[row][col] - offset) * scale) >> 4;
      pixelB[row][col] = ((pixelB[row][col] - offset) * scale) >> 4;
      pixelG[row][col] = ((pixelG[row][col] - offset) * scale) >> 4;
    }
  }
}

int main(void) {
  in_int_t offset;
  in_int_t scale;
  inout_int_t pixelR[N][N];
  inout_int_t pixelG[N][N];
  inout_int_t pixelB[N][N];

  srand(13);
  offset = rand() % 100;
  scale = rand() % 10;
  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      pixelR[y][x] = rand() % 1000;
      pixelG[y][x] = rand() % 1000;
      pixelB[y][x] = rand() % 1000;
    }
  }

  CALL_KERNEL(video_filter, offset, scale, pixelR, pixelG, pixelB);
  return 0;
}
