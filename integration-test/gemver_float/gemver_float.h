typedef float in_float_t;
typedef float inout_float_t;

#define N 10

void gemver_float(in_float_t alpha, in_float_t beta, inout_float_t A[N][N],
                  in_float_t u1[N], in_float_t v1[N], in_float_t u2[N],
                  in_float_t v2[N], inout_float_t w[N], inout_float_t x[N],
                  in_float_t y[N], in_float_t z[N]);
