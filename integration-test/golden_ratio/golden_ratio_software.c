#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#define ITERS 10
float golden_ratio(float x0) {
  float x = x0;
  for (int i = 0; i < ITERS; i++) {
    float original_x = x;
    while (true) {
      float next_x = 0.5f * (x + original_x / x);
      if (fabs(next_x - x) < 0.1f)
        break;
      x = next_x;
    }
    x += 1.0f;
  }
  return x - 1.0f;
}

int main(void) {
  float x0 = 100.0f;
  float result = golden_ratio(x0);
  printf("Golden Ratio: %f\n", result);
  return 0;
}