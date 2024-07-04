typedef float in_float_t;
typedef float inout_float_t;

#define N 30

float bicg_float(in_float_t A[N][N], inout_float_t s[N], inout_float_t q[N],
                 in_float_t p[N], in_float_t r[N]);
