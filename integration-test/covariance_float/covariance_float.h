#ifndef COVARIANCE_FLOAT_COVARIANCE_FLOAT_H
#define COVARIANCE_FLOAT_COVARIANCE_FLOAT_H
typedef float inout_float_t;
typedef float out_float_t;

void covariance_float(inout_float_t data[30][30], out_float_t symmat[30][30],
                      out_float_t mean[30]);
#endif // COVARIANCE_COVARIANCE_H
