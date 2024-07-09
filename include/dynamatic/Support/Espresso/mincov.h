#ifndef DYNAMATIC_SUPPORT_ESPRESSO_MINCOV_H
#define DYNAMATIC_SUPPORT_ESPRESSO_MINCOV_H

#include "dynamatic/Support/Espresso/sparse.h"

/* exported */
extern sm_row *sm_minimum_cover(sm_matrix *A, int *weight, int heuristic,
                                int debug_level);

#endif // DYNAMATIC_SUPPORT_ESPRESSO_MINCOV_H
