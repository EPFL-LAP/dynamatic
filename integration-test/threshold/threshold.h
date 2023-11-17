//===- threshold.h - Threshold on RGB channels --------------------*- C -*-===//
//
// Declares the threshold kernel which sets the RGB channels of a pixel color
// value to 0 if their sum doesn't exceed a threshold value.
//
//===----------------------------------------------------------------------===//

typedef int in_int_t;
typedef int inout_int_t;

#define N 1000

/// Sets the RGB channels of a pixel color value to 0 if their sum doesn't
/// exceed a threshold value.
void threshold(in_int_t th, inout_int_t red[N], inout_int_t green[N],
               inout_int_t blue[N]);