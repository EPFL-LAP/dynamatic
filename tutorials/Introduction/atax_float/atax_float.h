typedef float in_float_t;
typedef float out_float_t;
typedef float inout_float_t;

#define N 20

void atax_float(in_float_t A[N][N], in_float_t x[N], inout_float_t y[N],
                inout_float_t tmp[N]);
