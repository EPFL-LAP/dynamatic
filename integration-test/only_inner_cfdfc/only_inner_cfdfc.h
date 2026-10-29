#ifndef ONLY_INNER_CFDFC_H
#define ONLY_INNER_CFDFC_H

typedef int in_int_t;
typedef int out_int_t;
typedef int inout_int_t;

#define N 64
#define M 4

#define LOOP_BOUND 20

void innermost_cfdfc(out_int_t a[N][M], out_int_t c[N]);

#endif // ONLY_INNER_CFDFC_H
