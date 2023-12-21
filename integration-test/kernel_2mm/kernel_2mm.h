#define NI 10
#define NJ 10
#define NK 10
#define NL 10

typedef int in_int_t;
typedef int out_int_t;
typedef int inout_int_t;

void kernel_2mm(in_int_t alpha, in_int_t beta, inout_int_t tmp[NI][NJ],
                in_int_t A[NI][NK], in_int_t B[NK][NJ], in_int_t C[NK][NL],
                inout_int_t D[NI][NL]);
