#ifndef DYNAMATIC_SUPPORT_ESPRESSO_SIGNATURE_H
#define DYNAMATIC_SUPPORT_ESPRESSO_SIGNATURE_H

#include "dynamatic/Support/Espresso/espresso.h"

#if BPI == 16
#define ODD_MASK 0xaaaa
#define EVEN_MASK 0x5555
#else
#define ODD_MASK 0xaaaaaaaa
#define EVEN_MASK 0x55555555
#endif

#define POSITIVE 1
#define NEGATIVE 0

#define PRESENT 1
#define ABSENT 0

#define RAISED 2

typedef struct {
  int variable;
  int free_count;
} VAR;

/* black_white.c */ extern void setup_bw(pset_family R, pset c);
/* black_white.c */ extern void free_bw(void);
/* black_white.c */ extern int black_white(void);
/* black_white.c */ extern void split_list(pset_family R, int v);
/* black_white.c */ extern void merge_list(void);
/* black_white.c */ extern void print_bw(int size);
/* black_white.c */ extern void variable_list_alloc(int size);
/* black_white.c */ extern void variable_list_init(int reduced_c_free_count,
                                                   int *reduced_c_free_list);
/* black_white.c */ extern void variable_list_delete(int element);
/* black_white.c */ extern void variable_list_insert(int element);
/* black_white.c */ extern int variable_list_empty(void);
/* black_white.c */ extern void get_next_variable(int *pv, int *pphase,
                                                  pset_family R);
/* black_white.c */ extern void print_variable_list(void);
/* black_white.c */ extern void reset_black_list(void);
/* black_white.c */ extern void push_black_list(void);
/* black_white.c */ extern void pop_black_list(void);
/* canonical.c */ extern pset_family
find_canonical_cover(pset_family F1, pset_family D, pset_family R);
/* essentiality.c */ extern pset_family
etr_order(pset_family F, pset_family E, pset_family R, pset c, pset d);
/* essentiality.c */ extern void aux_etr_order(pset_family F, pset_family E,
                                               pset_family R, pset c, pset d);
/* essentiality.c */ extern pset_family get_mins(pset c);
/* essentiality.c */ extern int ascending(VAR *p1, VAR *p2);
/* util_signature.c */ extern void set_time_limit(int seconds);
/* util_signature.c */ extern void print_cover(pset_family F, char *name);
/* util_signature.c */ extern int sf_equal(pset_family F1, pset_family F2);
/* util_signature.c */ extern int mem_usage(char *name);
/* util_signature.c */ extern int time_usage(char *name);
/* util_signature.c */ extern void s_totals(long time, int i);
/* util_signature.c */ extern void s_runtime(long total);
/* sigma.c */ extern pset get_sigma(pset_family R, pset c);
/* sigma.c */ extern void set_not(pset c);
/* signature.c */ extern void cleanup(void);
/* signature.c */ extern pset_family signature(pset_family F1, pset_family D1,
                                               pset_family R1);
/* signature.c */ extern pset_family generate_primes(pset_family F,
                                                     pset_family R);
/* signature_exact.c */ extern pset_family
signature_minimize_exact(pset_family ESCubes, pset_family ESSet);
/* signature_exact.c */ extern sm_matrix *
signature_form_table(pset_family ESCubes, pset_family ESSet);

#endif // DYNAMATIC_SUPPORT_ESPRESSO_SIGNATURE_H
