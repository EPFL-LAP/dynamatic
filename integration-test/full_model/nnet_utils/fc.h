

#define DENSE_RELU_LAYER(INPUT, OUTPUT, INPUT_SZ, OUTPUT_SZ, W, B, ACC, TMP)    \
_Pragma("clang loop unroll(full)")                                              \
    for (int j = 0; j < OUTPUT_SZ; j++) {                                       \
        ACC = 0;                                                                \
        ACC = (dense_accum_t)(B[j] << (FRAC_DEFAULT));                          \
_Pragma("clang loop unroll(full)")                                              \
        for (int i = 0; i < INPUT_SZ; i++) {                                    \
            ACC += INPUT[i] * W[j][i];                                          \
        }                                                                       \
        /* TRUNCATE */                                                          \
        ACC = ACC >> (FRAC_DEFAULT);                                            \
        TMP = (default_t)ACC;                                                   \
        /* RELU ACTIVATION */                                                   \
        OUTPUT[j] = TMP > 0 ? TMP : 0;                                          \
    }

#define DENSE_LAYER(INPUT, OUTPUT, INPUT_SZ, OUTPUT_SZ, W, B, ACC, TMP)    \
_Pragma("clang loop unroll(full)")                                              \
    for (int j = 0; j < OUTPUT_SZ; j++) {                                       \
        ACC = 0;                                                                \
        ACC = (dense_accum_t)(B[j] << (FRAC_DEFAULT));                          \
_Pragma("clang loop unroll(full)")                                              \
        for (int i = 0; i < INPUT_SZ; i++) {                                    \
            ACC += INPUT[i] * W[j][i];                                          \
        }                                                                       \
        /* TRUNCATE */                                                          \
        ACC = ACC >> (FRAC_DEFAULT);                                            \
        OUTPUT[j] = (default_t)ACC;                                             \
    }

#define ARGMAX(INPUT, OUTPUT, INPUT_SZ, TMP_ARGMAX)                             \
    TMP_ARGMAX = -(1 << (NB_DEFAULT-1));                                        \
    for (int i = 0; i < INPUT_SZ; i++) {                                        \
        TMP_ARGMAX = (TMP_ARGMAX < INPUT[i]) ? INPUT[i] : TMP_ARGMAX;           \
    }                                                                           \
_Pragma("clang loop unroll(full)")                                              \
    for (int i = 0; i < INPUT_SZ; i++) {                                        \
        OUTPUT[i] = (TMP_ARGMAX == INPUT[i]) ? (1 << (FRAC_DEFAULT)) : 0;       \
    }