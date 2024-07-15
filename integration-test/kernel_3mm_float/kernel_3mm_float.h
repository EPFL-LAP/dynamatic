typedef float in_float_t;
typedef float out_float_t;
typedef float inout_float_t;

#define NI 10
#define NJ 10
#define NK 10
#define NL 10
#define NM 10
#define N 10

void kernel_3mm_float(in_float_t A[N][N], in_float_t B[N][N],
                      in_float_t C[N][N], in_float_t D[N][N],
                      inout_float_t E[N][N], inout_float_t F[N][N],
                      inout_float_t G[N][N]);
