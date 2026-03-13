#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

// https://www.boost.org/doc/libs/1_36_0/libs/numeric/ublas/doc/matrix.htm
// Types for representing / initializing matrices
using MatIntZero = boost::numeric::ublas::zero_matrix<int>;
using MatIntType = boost::numeric::ublas::matrix<int>;

// https://www.boost.org/doc/libs/latest/libs/numeric/ublas/doc/matrix_proxy.html#matrix_row
// Type for accessing / manipulating a row of a matrix
using MatIntRow = boost::numeric::ublas::matrix_row<MatIntType>;

// Performs Gaussian elimination on the matrix until the row echelon form is
// reached, while using some tricks to avoid numerical instability.
void gaussianElimination(MatIntType &m);
