typedef float in_float_t;
typedef float out_float_t;
typedef float inout_float_t;

void symm_float(in_float_t A[30][30], in_float_t B[30][30],
                inout_float_t C[30][30], in_float_t alpha, in_float_t beta);
