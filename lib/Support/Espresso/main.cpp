/*
 * Revision Control Information
 *
 * $Source$
 * $Author$
 * $Revision$
 * $Date$
 *
 */
/*
 *  Main driver for espresso
 *
 *  Old style -do xxx, -out xxx, etc. are still supported.
 */

// #include "dynamatic/Support/Espresso/espresso.h"
#include "dynamatic/Support/Espresso/main.h"
#include <time.h>
#include <unistd.h>

static int input_type = FD_type;

void getPLA(char *s, int option, pPLA *PLA, int out_type) {
  int needs_dcset, needs_offset;

  needs_dcset = TRUE;
  needs_offset = TRUE;
  read_pla(s, needs_dcset, needs_offset, input_type, PLA);

  (*PLA)->filename = 0;
  filename = (*PLA)->filename;
  /*    (void) fclose(fp);*/
  /* hackto support -Dmany */
}

void runtime(void) {
  int i;
  long total = 1, temp;

  for (i = 0; i < TIME_COUNT; i++) {
    total += total_time[i];
  }
  for (i = 0; i < TIME_COUNT; i++) {
    if (total_calls[i] != 0) {
      temp = 100 * total_time[i];
      printf("# %s\t%2d call(s) for %s (%2ld.%01ld%%)\n", total_name[i],
             total_calls[i], print_time(total_time[i]), temp / total,
             (10 * (temp % total)) / total);
    }
  }
}

void init_runtime(void) {
  total_name[READ_TIME] = "READ       ";
  total_name[WRITE_TIME] = "WRITE      ";
  total_name[GREDUCE_TIME] = "REDUCE_GASP";
  total_name[GEXPAND_TIME] = "EXPAND_GASP";
  total_name[GIRRED_TIME] = "IRRED_GASP ";
  total_name[MV_REDUCE_TIME] = "MV_REDUCE  ";
  total_name[RAISE_IN_TIME] = "RAISE_IN   ";
  total_name[VERIFY_TIME] = "VERIFY     ";
  total_name[PRIMES_TIME] = "PRIMES     ";
  total_name[MINCOV_TIME] = "MINCOV     ";
}

char *run_espresso(char *s) {
  int out_type, option;
  pPLA PLA, PLA1;
  pcover Fold;
  cost_t cost;
  Bool error;
  extern char *optarg;
  extern int optind;

  error = FALSE;
  init_runtime();
#ifdef RANDOM
  srandom(314973);
#endif

  option = 0;              /* default -D: ESPRESSO */
  out_type = EQNTOTT_type; /* default -o: default is ON-set only */
  debug = 0;               /* default -d: no debugging info */
  verbose_debug = FALSE;   /* default -v: not verbose */
  print_solution = TRUE;   /* default -x: print the solution (!) */
  summary = FALSE;         /* default -s: no summary */
  trace = FALSE;           /* default -t: no trace information */
  remove_essential = TRUE; /* default -e: */
  force_irredundant = TRUE;
  unwrap_onset = TRUE;
  single_expand = FALSE;
  pos = FALSE;
  recompute_onset = FALSE;
  use_super_gasp = FALSE;
  use_random_order = FALSE;
  kiss = FALSE;
  echo_comments = TRUE;
  echo_unknown_commands = TRUE;

  /* the remaining arguments are argv[optind ... argc-1] */
  PLA = PLA1 = NIL(PLA_t);

  getPLA(s, option, &PLA, out_type);

  if (PLA == NULL) {
    printf("Null PLA!\n");
    exit(1);
  }

  if (PLA->F == NULL) {
    printf("Null PLA->F!\n");
    exit(1);
  }

  Fold = sf_save(PLA->F);
  PLA->F = espresso(PLA->F, PLA->D, PLA->R);
  EXECUTE(error = verify(PLA->F, Fold, PLA->D), VERIFY_TIME, PLA->F, cost);
  if (error) {
    print_solution = FALSE;
    PLA->F = Fold;
    (void)check_consistency(PLA);
  } else {
    free_cover(Fold);
  }

  char *result = eqn_output(PLA);

  /* Crash and burn if there was a verify error */
  if (error) {
    fatal("cover verification failed");
  }

  /* cleanup all used memory */
  free_PLA(PLA);
  FREE(cube.part_size);
  setdown_cube(); /* free the cube/cdata structure data */
  sf_cleanup();   /* free unused set structures */
  sm_cleanup();   /* sparse matrix cleanup */

  return result;
}
