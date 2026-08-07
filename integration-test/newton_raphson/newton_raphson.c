//===- newton_raphson.c -------------------------------------------*- C -*-===//
//
// Copied from the description in FPGA '19 paper: This is a hybrid algorithm of
// bisection and the Newton-Raphson method for finding the roots of a function.
// The hybrid algorithm takes a bisection step whenever Newton-Raphson would
// take the solution out of bounds and therefore improves the convergence
// properties of the algorithm over the standard Newton-Raphson method. The
// algorithm contains a for loop with an if-else statement to determine which of
// the two methods to use for a particular data point. Static predication is
// limited by the complex if condition and, as the next loop iteration requires
// the data computed in the current one, it must be scheduled for after the
// condition has been determined
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "stdbool.h"
#include <stdlib.h>

int newton_raphson(int rts, int x1, int xh, int df) {
  int i = 0;
  int dx;
  while (i < 100) {
    int f = 2 * rts - 100;
    i++;
    if (f < 0)
      x1 = rts;
    else
      xh = rts;

    // clang-format off
    #pragma DYN speculate variable=complicatedIfElse max_predictions=4 style=standard
    // clang-format on
    bool complicatedIfElse = ((rts - x1) * df - f) * ((rts - xh) * df - f) <= 0;
    if (complicatedIfElse) {
      dx = (xh - x1) >> 2;
      rts = x1 + dx;
    } else {
      dx = f >> 2;
      rts -= dx;
    }
  }
  return rts;
}

int main(void) {
  int rts = 20;
  int x1 = -1000;
  int xh = 1000;
  int df = 2;
  CALL_KERNEL(newton_raphson, rts, x1, xh, df);
  return 0;
}
