#include "dynamatic/Integration.h"

#define N 8
#define M 4

double lapprox(const double lx[N], const double ly[N], double x) {
  for (int i = 1; i < N; i++) {
    if (lx[i] > x) {
      return ly[i - 1] +
             (ly[i] - ly[i - 1]) * (lx[i] - x) / (lx[i] - lx[i - 1]);
    }
  }
  return ly[N - 1];
}

int interpolate(const double xs[M], double ys[M], const double lx[N],
                const double ly[N]) {
  for (int i = 0; i < M; i++) {
    ys[i] = lapprox(lx, ly, xs[i]);
  }
  return 0;
}

int main(void) {
  const double lx[N] = {0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5};
  const double ly[N] = {0., 0., 1., 3., 3.3, 2.7, 2.5, 2.5};
  double xs[M] = {-1., 1.75, 1.1, 2.9};
  double ys[M];

  CALL_KERNEL(interpolate, xs, ys, lx, ly);

  return 0;
}
