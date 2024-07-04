typedef float in_float_t;
typedef float out_float_t;
typedef float inout_float_t;

void gesummv_float(in_float_t alpha, in_float_t beta, in_float_t A[30][30],
                   in_float_t B[30][30], out_float_t tmp[30], out_float_t y[30],
                   in_float_t x[30]);
