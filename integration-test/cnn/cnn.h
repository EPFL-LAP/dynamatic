#ifndef CNN_CNN_H
#define CNN_CNN_H

typedef int in_int_t;
typedef int out_int_t;
typedef int inout_int_t;

#define ParallelOut 2
#define ImSize 10
#define OutImSize 10
#define NumIn 10
#define NumOut 10
#define kKernel 4

// TODO: the parameters below can be defined in terms of the parameters above,
// but this is not yet supported in the legacy hls-verifier (it needs the array
// length in the square bracket to be an actual number.).
#define CSize 114
#define weightSize 43
#define inputSize 38
#define outputSize 120
#define biasSize 12

void cnn(in_int_t input[inputSize], in_int_t bias[biasSize],
         in_int_t weight[weightSize], inout_int_t C[CSize],
         out_int_t output[outputSize]);

#endif // CNN_CNN_H
