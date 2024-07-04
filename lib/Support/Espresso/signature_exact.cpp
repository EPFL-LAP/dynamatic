#include "dynamatic/Support/Espresso/espresso.h"
#include "dynamatic/Support/Espresso/mincov.h"
#include "dynamatic/Support/Espresso/signature.h"

/*
 * signature_minimize_exact:
 * What does it do: forms and solves the covering table whose rows are
 *	essential signature cubes (ESCubes) and whose columns are
 *	union of essential signature sets (ESSet)
 * Input:
 *	ESCubes: essential signature cubes
 *	ESSet: union of essential signature sets
 * Output:
 *	COVER: exact cover
 */

pcover signature_minimize_exact(pset_family ESCubes, pset_family ESSet) {
  pcube p;
  sm_matrix *table;
  sm_row *cover;
  sm_element *pe;
  pcover COVER;
  int index;
  int *weights, heur, level;

  /* number ESCubes, ESSet */
  foreachi_set(ESCubes, index, p) { PUTSIZE(p, index); }
  foreachi_set(ESSet, index, p) { PUTSIZE(p, index); }

  /* form the covering table */
  S_EXECUTE(table = signature_form_table(ESCubes, ESSet), MINCOV_TIME);

  /* solve the covering problem */
  weights = NIL(int);
  heur = FALSE;
  level = 0;
  S_EXECUTE(cover = sm_minimum_cover(table, weights, heur, level), MINCOV_TIME);

  /* form the cover */
  COVER = new_cover(100);
  sm_foreach_row_element(cover, pe) {
    COVER = sf_addset(COVER, GETSET(ESSet, pe->col_num));
  }

  sm_free(table);
  sm_row_free(cover);

  return COVER;
}

sm_matrix *signature_form_table(pset_family ESCubes, pset_family ESSet) {
  sm_matrix *table;
  int row, column;
  pcube c, p;

  table = sm_alloc();

  foreachi_set(ESSet, column, p) {
    if (column % 1000 == 0) {
      sm_col_dominance(table, NULL);
    }
    foreachi_set(ESCubes, row, c) {
      if (setp_implies(c, p)) {
        sm_insert(table, row, column);
      }
    }
  }
  sm_col_dominance(table, NULL);

  return table;
}
