#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using MatIntZero = boost::numeric::ublas::zero_matrix<int>;
using MatIntType = boost::numeric::ublas::matrix<int>;
using MatIntRow = boost::numeric::ublas::matrix_row<MatIntType>;
// Performs Gaussian elimination on the matrix until the row echelon form is
// reached, while using some tricks to avoid numerical instability.
void gaussianElimination(MatIntType &m);
