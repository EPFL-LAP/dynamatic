//===- admm.h - Computes ADMM kernel -----------------*- C -*-===//
//
// Declares the admm kernel which uses the Alternating Direction Method of
// Multipliers OQSP
//
//===----------------------------------------------------------------------===//

#ifndef ADMM_ADMM_H
#define ADMM_ADMM_H

typedef float in_float_t;
typedef float out_float_t;

/// Computes the Alternating Direction Method of Multipliers optimization method.
void admm(in_float_t vdc, in_float_t inp[30], in_float_t KKT_inv[30][30],
          out_float_t out[2], out_float_t x[30], out_float_t z[30],
          out_float_t y[30], out_float_t rhs[30], out_float_t temp_x_tilde[30]);

#endif // ADMM_ADMM_H
