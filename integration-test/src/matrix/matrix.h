//===- matrix.h - Matrix multiplication ---------------------------*- C -*-===//
//
// Declares the matrix kernel which computes a matrix multiplication.
//
//===----------------------------------------------------------------------===//

typedef int in_int_t;
typedef int out_int_t;

#define A_ROWS 30
#define A_COLS 30
#define B_ROWS A_COLS
#define B_COLS 30

/// Multiplies two matrices and stores the multiplication's result in the last
/// argument.
void matrix(in_int_t inA[A_ROWS][A_COLS], in_int_t inB[A_COLS][B_COLS],
            out_int_t outC[A_ROWS][B_COLS]);
