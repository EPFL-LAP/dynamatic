#ifndef KMP_KMP_H
#define KMP_KMP_H
typedef int in_int_t;
typedef int out_int_t;
#define PATTERN_SIZE 64
#define STRING_SIZE 200

void kmp(in_int_t pattern[PATTERN_SIZE],
        in_int_t input1[STRING_SIZE],
        in_int_t input2[STRING_SIZE],
        in_int_t input3[STRING_SIZE],
        in_int_t input4[STRING_SIZE],
        in_int_t input5[STRING_SIZE],
        in_int_t input6[STRING_SIZE],
        in_int_t input7[STRING_SIZE],
        in_int_t input8[STRING_SIZE],
        in_int_t input9[STRING_SIZE],
        in_int_t input10[STRING_SIZE],
        in_int_t kmpNext[PATTERN_SIZE],
        out_int_t results[10]);

#endif // KMP_KMP_H
