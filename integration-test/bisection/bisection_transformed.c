#include "dynamatic/Integration.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#define MAX_ITER 100
#define F(x) ((x) * (x) - 2.0f)
float bisection(float a, float b, float tol) {
  float fa = F(a);

  // Assume fa and fb have different signs

  int i = 0;
  float c, fc;
  do {
    while (true) {
      c = 0.5f * (a + b);
      fc = F(c);
      if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
        return c;
      }
      bool cond = fa * fc < 0;
      if (!cond)
        break;
      b = c;
      i++;
      if (i >= MAX_ITER)
        return 0; // Failed
    }
    a = c;
    fa = fc;
    i++;
  } while (i < MAX_ITER);

  return 0; // Failed
}

int main(void) {
  float a = 0.0f;
  float b = 100.0f;
  float tol = 1e-10f;

  CALL_KERNEL(bisection, a, b, tol);
  return 0;
}
