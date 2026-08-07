//===- ii_edge_cases.c - Loops exercising the II monitor's edges --*- C -*-===//
//
// Kernel whose loops exercise the edge cases of the '--instrument-ii' monitor:
// a loop that never runs (its monitor must stay silent), one that runs a
// single iteration (a single reported line, with no interval to it) and one
// that runs exactly two (the shortest activation with a measurable interval).
// The loop bounds are kernel arguments so that no loop can be folded away at
// compile time.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"

#define N 8

void ii_edge_cases(int n0, int n1, int n2, int a[N]) {
  // Never entered (n0 == 0).
  int sum0 = 0;
  for (int i = 0; i < n0; i++)
    sum0 += a[i];
  a[0] = sum0;

  // Runs a single iteration (n1 == 1).
  int sum1 = 0;
  for (int i = 0; i < n1; i++)
    sum1 += a[i] * a[i];
  a[1] = sum1;

  // Runs two iterations (n2 == 2).
  int sum2 = 0;
  for (int i = 0; i < n2; i++)
    sum2 += a[i] * sum2 + 1;
  a[2] = sum2;
}

int main(void) {
  int n0 = 0;
  int n1 = 1;
  int n2 = 2;
  int a[N];
  for (int i = 0; i < N; ++i)
    a[i] = i + 1;

  CALL_KERNEL(ii_edge_cases, n0, n1, n2, a);
  return 0;
}
