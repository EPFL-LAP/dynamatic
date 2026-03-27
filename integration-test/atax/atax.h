#ifndef ATAX_ATAX_FLOAT
#define ATAX_ATAX_FLOAT

typedef int in_int_t;
typedef int out_int_t;
typedef int inout_int_t;

#define N 20

void atax(in_int_t A[N][N], in_int_t x[N], inout_int_t y[N],
          inout_int_t tmp[N]);
#endif // ATAX_ATAX_FLOAT
