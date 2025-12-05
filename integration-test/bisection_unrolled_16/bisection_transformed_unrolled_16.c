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
  int i = 0;
  float fc;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop0;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop0;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop0:
  result0[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop1;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop1;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop1:
  result1[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop2;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop2;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop2:
  result2[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop3;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop3;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop3:
  result3[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop4;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop4;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop4:
  result4[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop5;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop5;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop5:
  result5[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop6;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop6;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop6:
  result6[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop7;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop7;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop7:
  result7[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop8;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop8;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop8:
  result8[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop9;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop9;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop9:
  result9[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop10;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop10;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop10:
  result10[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop11;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop11;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop11:
  result11[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop12;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop12;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop12:
  result12[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop13;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop13;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop13:
  result13[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop14;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop14;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop14:
  result14[0] = c;

  a = a0;
  b = b0;
  fa = F(a);
  c = 0.0f;
  i = 0;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        goto end_loop15;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER) {
        c = 0; // Failed
        goto end_loop15;
      }
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);
end_loop15:
  result15[0] = c;
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
