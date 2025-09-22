#include "dynamatic/Integration.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#define MAX_ITER 100
#define F(x) ((x) * (x) - 2.0f)
float bisection(float a, float b, float tol) {
  float fa = F(a);

  // Assume fa and fb have different signs

  for (int i = 0; i < MAX_ITER; i++) {
    float c = 0.5f * (a + b);
    float fc = F(c);
    if (fabs(fc) < tol || (b - a) / 2.0f < tol) {
      return c;
    }
    bool cond = fa * fc < 0;
    b = cond ? c : b;
    a = cond ? a : c;
    fa = cond ? fa : fc;
  }

  return 0; // Failed
}

int main(void) {
  float a = 0.0f;
  float b = 100.0f;
  float tol = 1e-10f;

  CALL_KERNEL(bisection, a, b, tol);
  return 0;
}
