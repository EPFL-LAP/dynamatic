#define NI 10
#define NJ 10
#define NK 10
#define NL 10
#define NM 10
#define N 10

typedef int in_int_t;
typedef int out_int_t;
typedef int inout_int_t;

void kernel_3mm(in_int_t A[NI][NK], in_int_t B[NK][NJ], in_int_t C[NJ][NM],
                in_int_t D[NM][NL], inout_int_t E[NI][NJ],
                inout_int_t F[NJ][NL], inout_int_t G[NI][NL]);