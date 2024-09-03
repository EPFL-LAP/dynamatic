#ifndef COVARIANCE_COVARIANCE_H
#define COVARIANCE_COVARIANCE_H

typedef int inout_int_t;
typedef int out_int_t;

void covariance(inout_int_t data[30][30], out_int_t symmat[30][30],
                out_int_t mean[30]);

#endif // COVARIANCE_COVARIANCE_H
