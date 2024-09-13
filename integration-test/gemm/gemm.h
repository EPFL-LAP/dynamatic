typedef int in_int_t;
typedef int out_int_t;
typedef int inout_int_t;

void gemm(in_int_t alpha, in_int_t beta, in_int_t A[30][30], in_int_t B[30][30],
          inout_int_t C[30][30]);
