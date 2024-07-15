typedef float in_float_t;
typedef float out_float_t;
typedef float inout_float_t;

#define NI 10
#define NJ 10
#define NK 10
#define NL 10
#define N 10

void kernel_2mm_float(in_float_t alpha, in_float_t beta,
                      inout_float_t tmp[N][N], in_float_t A[N][N],
                      in_float_t B[N][N], in_float_t C[N][N],
                      inout_float_t D[N][N]);
