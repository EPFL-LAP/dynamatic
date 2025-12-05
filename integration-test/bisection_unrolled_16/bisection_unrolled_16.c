#include "dynamatic/Integration.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#define MAX_ITER 10
#define F(x) ((x) * (x) - 2.0f)
void bisection_unrolled_16(float a0, float b0, float tol, float result0[2],
                           float result1[2], float result2[2], float result3[2],
                           float result4[2], float result5[2], float result6[2],
                           float result7[2], float result8[2], float result9[2],
                           float result10[2], float result11[2],
                           float result12[2], float result13[2],
                           float result14[2], float result15[2]) {
  float a = a0;
  float b = b0;
  float fa = F(a);
  float c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result0[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result1[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result2[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result3[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result4[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result5[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result6[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result7[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result8[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result9[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result10[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result11[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result12[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result13[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result14[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  for (int i = 0; i < MAX_ITER; i++) {
    c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      result15[0] = c;
      break;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }
}

int main(void) {
  float a0 = 0.0f;
  float b0 = 100.0f;

  float tol = 1e-10f;

  float result0[2] = {0};
  float result1[2] = {0};
  float result2[2] = {0};
  float result3[2] = {0};
  float result4[2] = {0};
  float result5[2] = {0};
  float result6[2] = {0};
  float result7[2] = {0};
  float result8[2] = {0};
  float result9[2] = {0};
  float result10[2] = {0};
  float result11[2] = {0};
  float result12[2] = {0};
  float result13[2] = {0};
  float result14[2] = {0};
  float result15[2] = {0};

  CALL_KERNEL(bisection_unrolled_16, a0, b0, tol, result0, result1, result2,
              result3, result4, result5, result6, result7, result8, result9,
              result10, result11, result12, result13, result14, result15);
  return 0;
}
