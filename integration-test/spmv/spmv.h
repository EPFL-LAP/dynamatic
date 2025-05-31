#ifndef SPMV_SPMV_H
#define SPMV_SPMV_H

typedef int in_int_t;
typedef int inout_int_t;

int spmv(in_int_t n, inout_int_t row[11], inout_int_t col[10],
         inout_int_t val[10], inout_int_t vec[10], inout_int_t out[10]);

#endif // SPMV_SPMV_H
