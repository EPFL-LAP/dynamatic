#include "kmp.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

int kmp(in_char_t pattern[PATTERN_SIZE], in_char_t input[STRING_SIZE],
        inout_int_t kmpNext[PATTERN_SIZE]) {
  int i, q;
  int n_matches = 0;

  int k;
  k = 0;
  kmpNext[0] = 0;

c1:
  for (q = 1; q < PATTERN_SIZE; q++) {
    char tmp = pattern[q];
  c2:
    while (k > 0 && pattern[k] != tmp) {
      k = kmpNext[q];
    }
    if (pattern[k] == tmp) {
      k++;
    }
    kmpNext[q] = k;
  }

  q = 0;
k1:
  for (i = 0; i < STRING_SIZE; i++) {
    char tmp = input[i];
  k2:
    while (q > 0 && pattern[q] != tmp) {
      q = kmpNext[q];
    }
    if (pattern[q] == tmp) {
      q++;
    }
    if (q >= PATTERN_SIZE) {
      n_matches++;
      q = kmpNext[q - 1];
    }
  }
  return n_matches;
}

int main(void) {
  in_char_t pattern[PATTERN_SIZE];
  in_char_t input[STRING_SIZE];
  inout_int_t kmpNext[PATTERN_SIZE];

  for (int i = 0; i < PATTERN_SIZE; ++i) {
    pattern[i] = rand() % 10;
    kmpNext[i] = rand() % 10;
    kmpNext[3] = 0;
  }
  for (int i = 0; i < STRING_SIZE; ++i) {
    input[i] = rand() % 10;
  }

  CALL_KERNEL(kmp, pattern, input, kmpNext);
}
