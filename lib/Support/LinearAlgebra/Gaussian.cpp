
#include "dynamatic/Support/LinearAlgebra/Gaussian.h"

void gaussianElimination(MatIntType &m) {
  size_t rows = m.size1();
  size_t cols = m.size2();

  // h: row index
  // k: leading non-zero position
  size_t h = 0, k = 0;
  while (h < rows && k < cols) {
    int pivotRow = -1;

    int pivotValue = std::numeric_limits<int>::max();

    // Find the row to pivot around
    for (size_t i = h; i < rows; ++i) {
      if (m(i, k) != 0) {
        if (std::abs(m(i, k)) < std::abs(pivotValue)) {
          pivotValue = m(i, k);
          pivotRow = i;
        }
      }
    }

    // no row with non-zero index at k -> look at next column
    if (pivotRow == -1) {
      ++k;
      continue;
    }

    MatIntRow pivot(m, pivotRow);
    MatIntRow other(m, h);
    pivot.swap(other);
    if (pivotValue < 0) {
      for (size_t i = k; i < cols; ++i) {
        m(h, i) *= -1;
      }
    }

    // eliminate other rows
    for (size_t i = h + 1; i < rows; ++i) {
      int factorPivot = m(h, k);
      int factorRow = m(i, k);
      MatIntRow pivot(m, h);
      MatIntRow eliminated(m, i);
      eliminated *= factorPivot;
      eliminated -= factorRow * pivot;
    }
    ++h;
    ++k;
  }
}
