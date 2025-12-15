#include "kmp.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>
#include <stdbool.h>

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
        out_int_t results[10]) {

  int q1 = 0, n1 = 0;

  int q2 = 0, n2 = 0;

  int q3 = 0, n3 = 0;

  int q4 = 0, n4 = 0;

  int q5 = 0, n5 = 0;

  int q6 = 0, n6 = 0;

  int q7 = 0, n7 = 0;

  int q8 = 0, n8 = 0;

  int q9 = 0, n9 = 0;

  int q10 = 0, n10 = 0;

  for (int i = 0; i < STRING_SIZE; i++) {
    while (q1 > 0 && pattern[q1] != input1[i]) {
      q1 = kmpNext[q1 - 1];
    }

    while (q2 > 0 && pattern[q2] != input2[i]) {
      q2 = kmpNext[q2 - 1];
    }

    while (q3 > 0 && pattern[q3] != input3[i]) {
      q3 = kmpNext[q3 - 1];
    }

    while (q4 > 0 && pattern[q4] != input4[i]) {
      q4 = kmpNext[q4 - 1];
    }

    while (q5 > 0 && pattern[q5] != input5[i]) {
      q5 = kmpNext[q5 - 1];
    }

    while (q6 > 0 && pattern[q6] != input6[i]) {
      q6 = kmpNext[q6 - 1];
    }

    while (q7 > 0 && pattern[q7] != input7[i]) {
      q7 = kmpNext[q7 - 1];
    }

    while (q8 > 0 && pattern[q8] != input8[i]) {
      q8 = kmpNext[q8 - 1];
    }

    while (q9 > 0 && pattern[q9] != input9[i]) {
      q9 = kmpNext[q9 - 1];
    }

    while (q10 > 0 && pattern[q10] != input10[i]) {
      q10 = kmpNext[q10 - 1];
    }

    if (pattern[q1] == input1[i])
      q1 = q1 + 1;
    if (q1 >= PATTERN_SIZE) {
      n1 = n1 + 1;
      q1 = kmpNext[q1 - 1];
    }


    if (pattern[q2] == input2[i])
      q2 = q2 + 1;
    if (q2 >= PATTERN_SIZE) {
      n2 = n2 + 1;
      q2 = kmpNext[q2 - 1];
    }


    if (pattern[q3] == input3[i])
      q3 = q3 + 1;
    if (q3 >= PATTERN_SIZE) {
      n3 = n3 + 1;
      q3 = kmpNext[q3 - 1];
    }


    if (pattern[q4] == input4[i])
      q4 = q4 + 1;
    if (q4 >= PATTERN_SIZE) {
      n4 = n4 + 1;
      q4 = kmpNext[q4 - 1];
    }


    if (pattern[q5] == input5[i])
      q5 = q5 + 1;
    if (q5 >= PATTERN_SIZE) {
      n5 = n5 + 1;
      q5 = kmpNext[q5 - 1];
    }


    if (pattern[q6] == input6[i])
      q6 = q6 + 1;
    if (q6 >= PATTERN_SIZE) {
      n6 = n6 + 1;
      q6 = kmpNext[q6 - 1];
    }


    if (pattern[q7] == input7[i])
      q7 = q7 + 1;
    if (q7 >= PATTERN_SIZE) {
      n7 = n7 + 1;
      q7 = kmpNext[q7 - 1];
    }


    if (pattern[q8] == input8[i])
      q8 = q8 + 1;
    if (q8 >= PATTERN_SIZE) {
      n8 = n8 + 1;
      q8 = kmpNext[q8 - 1];
    }


    if (pattern[q9] == input9[i])
      q9 = q9 + 1;
    if (q9 >= PATTERN_SIZE) {
      n9 = n9 + 1;
      q9 = kmpNext[q9 - 1];
    }


    if (pattern[q10] == input10[i])
      q10 = q10 + 1;
    if (q10 >= PATTERN_SIZE) {
      n10 = n10 + 1;
      q10 = kmpNext[q10 - 1];
    }


  }
  results[0] = n1;
  results[1] = n2;
  results[2] = n3;
  results[3] = n4;
  results[4] = n5;
  results[5] = n6;
  results[6] = n7;
  results[7] = n8;
  results[8] = n9;
  results[9] = n10;

}
int main(void) {
  in_int_t pattern[PATTERN_SIZE];
  for (int i = 0; i < PATTERN_SIZE; ++i)
    pattern[i] = i % 2;

  in_int_t input1[STRING_SIZE];
  in_int_t input2[STRING_SIZE];
  in_int_t input3[STRING_SIZE];
  in_int_t input4[STRING_SIZE];
  in_int_t input5[STRING_SIZE];
  in_int_t input6[STRING_SIZE];
  in_int_t input7[STRING_SIZE];
  in_int_t input8[STRING_SIZE];
  in_int_t input9[STRING_SIZE];
  in_int_t input10[STRING_SIZE];

  in_int_t kmpNext[PATTERN_SIZE];
  out_int_t results[10];

  int flip;
  srand(2);
  for (int i = 0; i < STRING_SIZE; ++i) {
    int ideal = i % 2;
    flip = (rand() % 10 == 0);
    input1[i] = flip ? 1 - ideal : ideal;
    flip = (rand() % 10 == 0);
    input2[i] = flip ? 1 - ideal : ideal;
    flip = (rand() % 10 == 0);
    input3[i] = flip ? 1 - ideal : ideal;
    flip = (rand() % 10 == 0);
    input4[i] = flip ? 1 - ideal : ideal;
    flip = (rand() % 10 == 0);
    input5[i] = flip ? 1 - ideal : ideal;
    flip = (rand() % 10 == 0);
    input6[i] = flip ? 1 - ideal : ideal;
    flip = (rand() % 10 == 0);
    input7[i] = flip ? 1 - ideal : ideal;
    flip = (rand() % 10 == 0);
    input8[i] = flip ? 1 - ideal : ideal;
    flip = (rand() % 10 == 0);
    input9[i] = flip ? 1 - ideal : ideal;
    flip = (rand() % 10 == 0);
    input10[i] = flip ? 1 - ideal : ideal;
  }

  int k = 0;
  kmpNext[0] = 0;
  for (int q = 1; q < PATTERN_SIZE; q++) {
    while (k > 0 && pattern[k] != pattern[q])
      k = kmpNext[k - 1];
    if (pattern[k] == pattern[q])
      k++;
    kmpNext[q] = k;
  }

  CALL_KERNEL(kmp, pattern, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, kmpNext, results);
}