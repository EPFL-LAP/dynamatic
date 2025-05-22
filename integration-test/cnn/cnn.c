//===- cnn.c - One layer in CNN -----------------------------------*- C -*===//
//
// Source:
// https://github.com/UCLA-VAST/AutoBridge/tree/master/archive/benchmarks/CNN
//
//===---------------------------------------------------------------------===//

#include "cnn.h"
#include "dynamatic/Integration.h"

// The inner loops, they are written in this way to make manual unrolling
// easier
#define PO_INIT(i, h, w, po)                                                   \
  index_c = (po) * ParallelOut + (h) * ImSize + (w);                           \
  C[index_c] = bias[((i) << 1) + (po)];

#define PO_CONV(i, j, h, w, po)                                                \
  for (int p = 0; p < kKernel; ++p) {                                          \
    for (int q = 0; q < kKernel; ++q) {                                        \
      index_weight = (i) + (po) + (j) + (q) + (p);                             \
      index_input = (j) + (h) + (p) + (w) + (q);                               \
      index_c = (po) * ParallelOut + (h) * ImSize + (w);                       \
      C[index_c] += weight[index_weight] * input[index_input];                 \
    }                                                                          \
  }

#define PO_RELU(i, h, w, po)                                                   \
  index_c = (po) * ParallelOut + (h) * OutImSize + (w);                        \
  tmp1 = C[index_c];                                                           \
  if (tmp1 > 0)                                                                \
    tmp2 = tmp1;                                                               \
  else                                                                         \
    tmp2 = 0;                                                                  \
  index_out = (i) * (NumOut / ParallelOut) + (h) * OutImSize + (w);            \
  output[index_out] = tmp2;

void cnn(in_int_t input[inputSize], in_int_t bias[biasSize],
         in_int_t weight[weightSize], inout_int_t C[CSize],
         out_int_t output[outputSize]) {
  int index_weight, index_input, index_c = 0, index_out;
  int tmp1, tmp2;
  for (int i = 0; i < NumOut / ParallelOut; ++i) {
    // Initialization
    for (int h = 0; h < ImSize; ++h) {
      for (int w = 0; w < ImSize; ++w) {
        for (int po = 0; po < ParallelOut; ++po) {
          PO_INIT(i, h, w, po)
        }
      }
    }

    // Convolution
    for (int j = 0; j < NumIn; ++j) {
      for (int h = 0; h < ImSize; ++h) {
        for (int w = 0; w < ImSize; ++w) {
          for (int po = 0; po < ParallelOut; ++po) {
            PO_CONV(i, j, h, w, po)
          }
        }
      }
    }

    // ReLU + Max pooling
    for (int h = 0; h < OutImSize; ++h) {
      for (int w = 0; w < OutImSize; ++w) {
        for (int po = 0; po < ParallelOut; ++po) {
          PO_RELU(i, h, w, po)
        }
      }
    }
  }
}

int main(void) {
  in_int_t input[inputSize];
  in_int_t bias[biasSize];
  in_int_t weight[weightSize];
  inout_int_t C[CSize];
  out_int_t output[outputSize];
  for (int i = 0; i < inputSize; ++i) {
    input[i] = (i << 1) + ImSize;
  }
  for (int i = 0; i < biasSize; ++i) {
    bias[i] = (i << 1) - ParallelOut;
  }
  for (int i = 0; i < weightSize; ++i) {
    weight[i] = (i << 1) - i;
  }
  for (int i = 0; i < CSize; ++i) {
    C[i] = 0;
  }

  CALL_KERNEL(cnn, input, bias, weight, C, output);
  return 0;
}
