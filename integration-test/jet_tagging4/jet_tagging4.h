#ifndef JET_TAGGING2_JET_TAGGING2_H
#define JET_TAGGING2_JET_TAGGING2_H


#include <stdint.h>

// typedef int in_int_t;
// typedef int out_int_t;
// typedef int inout_int_t;

// hls-fpga-machine-learning insert layer precision
#define NB_DEFAULT 16
#define INT_DEFAULT 6
#define FRAC_DEFAULT NB_DEFAULT - INT_DEFAULT

#define NB_ACC  64
#define INT_ACC 44
#define FRAC_ACC NB_ACC - INT_ACC

typedef int64_t dense_accum_t;
typedef int16_t default_t;

// hls-fpga-machine-learning insert layers

// Input dimensions
#define INPUT_SIZE 16

// Layer 1 dimensions
#define IN_L0 INPUT_SIZE
#define OUT_L0 64 

// Layer 2 dimensions
#define IN_L1 OUT_L0
#define OUT_L1 32 

// Layer 3 dimensions
#define IN_L2 OUT_L1
#define OUT_L2 32 

// Layer 4 dimensions
#define IN_L3 OUT_L2
#define OUT_L3 5 


#endif // JET_TAGGING2_JET_TAGGING2_H