#ifndef HISTOGRAM_2_HISTOGRAM_2_H
#define HISTOGRAM_2_HISTOGRAM_2_H

typedef int in_int_t;
typedef float in_float_t;
typedef float inout_float_t;

void histogram_2(in_int_t feature[1000], in_float_t weight[1000],
  inout_float_t hist[1000], in_int_t n);

#endif // HISTOGRAM_2_HISTOGRAM_2_H
