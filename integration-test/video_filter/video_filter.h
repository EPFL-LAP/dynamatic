//===- video_filter.h - RGB color transformations -----------------*- C -*-===//
//
// Declares the video_filter kernel which performs some simple transformations
// on the RGB channels of a pixels' color values.
//
//===----------------------------------------------------------------------===//

typedef int in_int_t;
typedef int inout_int_t;

#define N 30

/// Applies a simple transformation (add offset, rescale, and shift) on each
/// channel of pixels' color values. Arrays are updated in place.
void video_filter(in_int_t offset, in_int_t scale, inout_int_t pixelR[N][N],
                  inout_int_t pixelG[N][N], inout_int_t pixelB[N][N]);