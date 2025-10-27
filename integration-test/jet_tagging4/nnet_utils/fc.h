

#define DENSE_RELU_LAYER(input, output, input_sz, output_sz, w, b, acc, tmp)    \
_Pragma("clang loop unroll(full)")                                              \
    for (int j = 0; j < output_sz; j++) {                                       \
        acc = 0;                                                                \
        acc = (dense_accum_t)(b[j] << (FRAC_DEFAULT));                          \
_Pragma("clang loop unroll(full)")                                              \
        for (int i = 0; i < input_sz; i++) {                                    \
            acc += input[i] * w[j][i];                                          \
        }                                                                       \
        /* TRUNCATE */                                                          \
        acc = acc >> (FRAC_DEFAULT);                                            \
        tmp = (common_t) acc;                                                   \
        /* RELU ACTIVATION */                                                   \
        output[j] = tmp > 0 ? tmp : 0;                                          \
    }