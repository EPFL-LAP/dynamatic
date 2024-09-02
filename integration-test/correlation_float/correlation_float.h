#ifndef CORRELATION_FLOAT_CORRELATION_FLOAT_H
#define CORRELATION_FLOAT_CORRELATION_FLOAT_H

typedef float inout_float_t;
typedef float out_float_t;
typedef float in_float_t;

void correlation_float(inout_float_t data[30][30], inout_float_t mean[30],
                       out_float_t symmat[30][30], inout_float_t stddev[30],
                       in_float_t float_n);
#endif // CORRELATION_FLOAT_CORRELATION_FLOAT_H
