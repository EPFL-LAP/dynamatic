#ifndef KMP_KMP_H
#define KMP_KMP_H
typedef char in_char_t;
typedef int inout_int_t;
#define PATTERN_SIZE 4
#define STRING_SIZE 1000

int kmp(in_char_t pattern[PATTERN_SIZE], in_char_t input[STRING_SIZE],
        inout_int_t kmpNext[PATTERN_SIZE]);
#endif // KMP_KMP_H
