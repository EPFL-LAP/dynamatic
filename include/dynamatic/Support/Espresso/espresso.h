/*
 *  espresso.h -- header file for Espresso-mv
 */

#ifndef DYNAMATIC_SUPPORT_ESPRESSO_ESPRESSO_H
#define DYNAMATIC_SUPPORT_ESPRESSO_ESPRESSO_H

#include "dynamatic/Support/Espresso/port.h"
#include "dynamatic/Support/Espresso/sparse.h"
#include "dynamatic/Support/Espresso/utility.h"

#define ptime() util_cpu_time()
#define print_time(t) util_print_time(t)

#undef NO_INLINE

/*-----THIS USED TO BE set.h----- */

/*
 *  set.h -- definitions for packed arrays of bits
 *
 *   This header file describes the data structures which comprise a
 *   facility for efficiently implementing packed arrays of bits
 *   (otherwise known as sets, cf. Pascal).
 *
 *   A set is a vector of bits and is implemented here as an array of
 *   unsigned integers.  The low order bits of set[0] give the index of
 *   the last word of set data.  The higher order bits of set[0] are
 *   used to store data associated with the set.  The set data is
 *   contained in elements set[1] ... set[LOOP(set)] as a packed bit
 *   array.
 *
 *   A family of sets is a two-dimensional matrix of bits and is
 *   implemented with the data type "set_family".
 *
 *   BPI == 32 and BPI == 16 have been tested and work.
 */

/* Define host machine characteristics of "unsigned int" */
#ifndef BPI
#define BPI 32 /* # bits per integer */
#endif

#if BPI == 32
#define LOGBPI 5 /* log(BPI)/log(2) */
#else
#define LOGBPI 4 /* log(BPI)/log(2) */
#endif

/* Define the set type */
typedef unsigned int *pset;

/* Define the set family type -- an array of sets */
typedef struct set_family {
  int wsize;               /* Size of each set in 'ints' */
  int sf_size;             /* User declared set size */
  int capacity;            /* Number of sets allocated */
  int count;               /* The number of sets in the family */
  int active_count;        /* Number of "active" sets */
  pset data;               /* Pointer to the set data */
  struct set_family *next; /* For garbage collection */
} set_family_t, *pset_family;

/* Macros to set and test single elements */
#define WHICH_WORD(element) (((element) >> LOGBPI) + 1)
#define WHICH_BIT(element) ((element) & (BPI - 1))

/* # of ints needed to allocate a set with "size" elements */
#if BPI == 32
#define SET_SIZE(size) ((size) <= BPI ? 2 : (WHICH_WORD((size) - 1) + 1))
#else
#define SET_SIZE(size) ((size) <= BPI ? 3 : (WHICH_WORD((size) - 1) + 2))
#endif

/*
 *  Three fields are maintained in the first word of the set
 *      LOOP is the index of the last word used for set data
 *      LOOPCOPY is the index of the last word in the set
 *      SIZE is available for general use (e.g., recording # elements in set)
 *      NELEM retrieves the number of elements in the set
 */
#define LOOP(set) (set[0] & 0x03ff)
#define PUTLOOP(set, i) (set[0] &= ~0x03ff, set[0] |= (i))
#if BPI == 32
#define LOOPCOPY(set) LOOP(set)
#define SIZE(set) (set[0] >> 16)
#define PUTSIZE(set, size) (set[0] &= 0xffff, set[0] |= ((size) << 16))
#else
#define LOOPCOPY(set) (LOOP(set) + 1)
#define SIZE(set) (set[LOOP(set) + 1])
#define PUTSIZE(set, size) ((set[LOOP(set) + 1]) = (size))
#endif

#define NELEM(set) (BPI * LOOP(set))
#define LOOPINIT(size) ((size <= BPI) ? 1 : WHICH_WORD((size) - 1))

/*
 *      FLAGS store general information about the set
 */
#define SET(set, flag) (set[0] |= (flag))
#define RESET(set, flag) (set[0] &= ~(flag))
#define TESTP(set, flag) (set[0] & (flag))

/* Flag definitions are ... */
#define PRIME 0x8000    /* cube is prime */
#define NONESSEN 0x4000 /* cube cannot be essential prime */
#define ACTIVE 0x2000   /* cube is still active */
#define REDUND 0x1000   /* cube is redundant(at this point) */
#define COVERED 0x0800  /* cube has been covered */
#define RELESSEN 0x0400 /* cube is relatively essential */

/* Most efficient way to look at all members of a set family */
#define foreach_set(R, last, p)                                                \
  for (p = R->data, last = p + R->count * R->wsize; p < last; p += R->wsize)
#define foreach_remaining_set(R, last, pfirst, p)                              \
  for (p = pfirst + R->wsize, last = R->data + R->count * R->wsize; p < last;  \
       p += R->wsize)
#define foreach_active_set(R, last, p)                                         \
  foreach_set(R, last, p) if (TESTP(p, ACTIVE))

/* Another way that also keeps the index of the current set member in i */
#define foreachi_set(R, i, p)                                                  \
  for (p = R->data, i = 0; i < R->count; p += R->wsize, i++)
#define foreachi_active_set(R, i, p) foreachi_set(R, i, p) if (TESTP(p, ACTIVE))

/* Looping over all elements in a set:
 *      foreach_set_element(pset p, int i, unsigned val, int base) {
 *		.
 *		.
 *		.
 *      }
 */
#define foreach_set_element(p, i, val, base)                                   \
  for (i = LOOP(p); i > 0;)                                                    \
    for (val = p[i], base = --i << LOGBPI; val != 0; base++, val >>= 1)        \
      if (val & 1)

/* Return a pointer to a given member of a set family */
#define GETSET(family, index) ((family)->data + (family)->wsize * (index))

/* Allocate and deallocate sets */
#define set_new(size) set_clear(ALLOC(unsigned int, SET_SIZE(size)), size)
#define set_full(size) set_fill(ALLOC(unsigned int, SET_SIZE(size)), size)
#define set_save(r) set_copy(ALLOC(unsigned int, SET_SIZE(NELEM(r))), r)
#define set_free(r) FREE(r)

/* Check for set membership, remove set element and insert set element */
#define is_in_set(set, e) (set[WHICH_WORD(e)] & (1 << WHICH_BIT(e)))
#define set_remove(set, e) (set[WHICH_WORD(e)] &= ~(1 << WHICH_BIT(e)))
#define set_insert(set, e) (set[WHICH_WORD(e)] |= 1 << WHICH_BIT(e))

/* Inline code substitution for those places that REALLY need it on a VAX */
#ifdef NO_INLINE
#define INLINEset_copy(r, a) (void)set_copy(r, a)
#define INLINEset_clear(r, size) (void)set_clear(r, size)
#define INLINEset_fill(r, size) (void)set_fill(r, size)
#define INLINEset_and(r, a, b) (void)set_and(r, a, b)
#define INLINEset_or(r, a, b) (void)set_or(r, a, b)
#define INLINEset_diff(r, a, b) (void)set_diff(r, a, b)
#define INLINEset_ndiff(r, a, b, f) (void)set_ndiff(r, a, b, f)
#define INLINEset_xor(r, a, b) (void)set_xor(r, a, b)
#define INLINEset_xnor(r, a, b, f) (void)set_xnor(r, a, b, f)
#define INLINEset_merge(r, a, b, mask) (void)set_merge(r, a, b, mask)
#define INLINEsetp_implies(a, b, when_false)                                   \
  if (!setp_implies(a, b))                                                     \
  when_false
#define INLINEsetp_disjoint(a, b, when_false)                                  \
  if (!setp_disjoint(a, b))                                                    \
  when_false
#define INLINEsetp_equal(a, b, when_false)                                     \
  if (!setp_equal(a, b))                                                       \
  when_false

#else

#define INLINEset_copy(r, a)                                                   \
  {                                                                            \
    int i_ = LOOPCOPY(a);                                                      \
    do                                                                         \
      r[i_] = a[i_];                                                           \
    while (--i_ >= 0);                                                         \
  }
#define INLINEset_clear(r, size)                                               \
  {                                                                            \
    int i_ = LOOPINIT(size);                                                   \
    *r = i_;                                                                   \
    do                                                                         \
      r[i_] = 0;                                                               \
    while (--i_ > 0);                                                          \
  }
#define INLINEset_fill(r, size)                                                \
  {                                                                            \
    int i_ = LOOPINIT(size);                                                   \
    *r = i_;                                                                   \
    r[i_] = ((unsigned int)(~0)) >> (i_ * BPI - size);                         \
    while (--i_ > 0)                                                           \
      r[i_] = ~0;                                                              \
  }
#define INLINEset_and(r, a, b)                                                 \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    PUTLOOP(r, i_);                                                            \
    do                                                                         \
      r[i_] = a[i_] & b[i_];                                                   \
    while (--i_ > 0);                                                          \
  }
#define INLINEset_or(r, a, b)                                                  \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    PUTLOOP(r, i_);                                                            \
    do                                                                         \
      r[i_] = a[i_] | b[i_];                                                   \
    while (--i_ > 0);                                                          \
  }
#define INLINEset_diff(r, a, b)                                                \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    PUTLOOP(r, i_);                                                            \
    do                                                                         \
      r[i_] = a[i_] & ~b[i_];                                                  \
    while (--i_ > 0);                                                          \
  }
#define INLINEset_ndiff(r, a, b, fullset)                                      \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    PUTLOOP(r, i_);                                                            \
    do                                                                         \
      r[i_] = fullset[i_] & (a[i_] | ~b[i_]);                                  \
    while (--i_ > 0);                                                          \
  }
#ifdef IBM_WATC
#define INLINEset_xor(r, a, b) (void)set_xor(r, a, b)
#define INLINEset_xnor(r, a, b, f) (void)set_xnor(r, a, b, f)
#else
#define INLINEset_xor(r, a, b)                                                 \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    PUTLOOP(r, i_);                                                            \
    do                                                                         \
      r[i_] = a[i_] ^ b[i_];                                                   \
    while (--i_ > 0);                                                          \
  }
#define INLINEset_xnor(r, a, b, fullset)                                       \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    PUTLOOP(r, i_);                                                            \
    do                                                                         \
      r[i_] = fullset[i_] & ~(a[i_] ^ b[i_]);                                  \
    while (--i_ > 0);                                                          \
  }
#endif
#define INLINEset_merge(r, a, b, mask)                                         \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    PUTLOOP(r, i_);                                                            \
    do                                                                         \
      r[i_] = (a[i_] & mask[i_]) | (b[i_] & ~mask[i_]);                        \
    while (--i_ > 0);                                                          \
  }
#define INLINEsetp_implies(a, b, when_false)                                   \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    do                                                                         \
      if (a[i_] & ~b[i_])                                                      \
        break;                                                                 \
    while (--i_ > 0);                                                          \
    if (i_ != 0)                                                               \
      when_false;                                                              \
  }
#define INLINEsetp_disjoint(a, b, when_false)                                  \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    do                                                                         \
      if (a[i_] & b[i_])                                                       \
        break;                                                                 \
    while (--i_ > 0);                                                          \
    if (i_ != 0)                                                               \
      when_false;                                                              \
  }
#define INLINEsetp_equal(a, b, when_false)                                     \
  {                                                                            \
    int i_ = LOOP(a);                                                          \
    do                                                                         \
      if (a[i_] != b[i_])                                                      \
        break;                                                                 \
    while (--i_ > 0);                                                          \
    if (i_ != 0)                                                               \
      when_false;                                                              \
  }

#endif

#if BPI == 32
#define count_ones(v)                                                          \
  (bit_count[v & 255] + bit_count[(v >> 8) & 255] +                            \
   bit_count[(v >> 16) & 255] + bit_count[(v >> 24) & 255])
#else
#define count_ones(v) (bit_count[v & 255] + bit_count[(v >> 8) & 255])
#endif

/* Table for efficient bit counting */
extern int bit_count[256];
/*----- END OF set.h ----- */

/* Define a boolean type */
#define Bool int
#define FALSE 0
#define TRUE 1
#define MAYBE 2
#define print_bool(x) ((x) == 0 ? "FALSE" : ((x) == 1 ? "TRUE" : "MAYBE"))

/* Map many cube/cover types/routines into equivalent set types/routines */
#define pcube pset
#define new_cube() set_new(cube.size)
#define free_cube(r) set_free(r)
#define pcover pset_family
#define new_cover(i) sf_new(i, cube.size)
#define free_cover(r) sf_free(r)
#define free_cubelist(T)                                                       \
  FREE(T[0]);                                                                  \
  FREE(T);

/* cost_t describes the cost of a cover */
typedef struct cost_struct {
  int cubes;  /* number of cubes in the cover */
  int in;     /* transistor count, binary-valued variables */
  int out;    /* transistor count, output part */
  int mv;     /* transistor count, multiple-valued vars */
  int total;  /* total number of transistors */
  int primes; /* number of prime cubes */
} cost_t, *pcost;

/* pair_t describes bit-paired variables */
typedef struct pair_struct {
  int cnt;
  int *var1;
  int *var2;
} pair_t, *ppair;

/* symbolic_list_t describes a single ".symbolic" line */
typedef struct symbolic_list_struct {
  int variable;
  int pos;
  struct symbolic_list_struct *next;
} symbolic_list_t;

/* symbolic_list_t describes a single ".symbolic" line */
typedef struct symbolic_label_struct {
  char *label;
  struct symbolic_label_struct *next;
} symbolic_label_t;

/* symbolic_t describes a linked list of ".symbolic" lines */
typedef struct symbolic_struct {
  symbolic_list_t *symbolic_list;   /* linked list of items */
  int symbolic_list_length;         /* length of symbolic_list list */
  symbolic_label_t *symbolic_label; /* linked list of new names */
  int symbolic_label_length;        /* length of symbolic_label list */
  struct symbolic_struct *next;
} symbolic_t;

/* PLA_t stores the logical representation of a PLA */
typedef struct {
  pcover F, D, R;              /* on-set, off-set and dc-set */
  char *filename;              /* filename */
  int pla_type;                /* logical PLA format */
  pcube phase;                 /* phase to split into on-set and off-set */
  ppair pair;                  /* how to pair variables */
  char **label;                /* labels for the columns */
  symbolic_t *symbolic;        /* allow binary->symbolic mapping */
  symbolic_t *symbolic_output; /* allow symbolic output mapping */
} PLA_t, *pPLA;

#define equals(a, b) (strcmp(a, b) == 0)

/* This is a hack which I wish I hadn't done, but too painful to change */
#define CUBELISTSIZE(T) (((pcube *)T[1] - T) - 3)

/* For documentation purposes */
#define IN
#define OUT
#define INOUT

/* The pla_type field describes the input and output format of the PLA */
#define F_type 1
#define D_type 2
#define R_type 4
#define PLEASURE_type 8               /* output format */
#define EQNTOTT_type 16               /* output format algebraic eqns */
#define KISS_type 128                 /* output format kiss */
#define CONSTRAINTS_type 256          /* output the constraints (numeric) */
#define SYMBOLIC_CONSTRAINTS_type 512 /* output the constraints (symbolic) */
#define FD_type (F_type | D_type)
#define FR_type (F_type | R_type)
#define DR_type (D_type | R_type)
#define FDR_type (F_type | D_type | R_type)

/* Definitions for the debug variable */
#define COMPL 0x0001
#define ESSEN 0x0002
#define EXPAND 0x0004
#define EXPAND1 0x0008
#define GASP 0x0010
#define IRRED 0x0020
#define REDUCE 0x0040
#define REDUCE1 0x0080
#define SPARSE 0x0100
#define TAUT 0x0200
#define EXACT 0x0400
#define MINCOV 0x0800
#define MINCOV1 0x1000
#define SHARP 0x2000
#define IRRED1 0x4000

#define VERSION "UC Berkeley, Espresso Version #2.3, Release date 01/31/88"

/* Define constants used for recording program statistics */
#define TIME_COUNT 22
#define READ_TIME 0
#define COMPL_TIME 1
#define ONSET_TIME 2
#define ESSEN_TIME 3
#define EXPAND_TIME 4
#define IRRED_TIME 5
#define REDUCE_TIME 6
#define GEXPAND_TIME 7
#define GIRRED_TIME 8
#define GREDUCE_TIME 9
#define PRIMES_TIME 10
#define MINCOV_TIME 11
#define MV_REDUCE_TIME 12
#define RAISE_IN_TIME 13
#define VERIFY_TIME 14
#define WRITE_TIME 15
#define FCC_TIME 16
#define ETR_TIME 17
#define ETRAUX_TIME 18
#define SIGMA_TIME 19
#define UCOMP_TIME 20
#define BW_TIME 21

/* For those who like to think about PLAs, macros to get at inputs/outputs */
#define NUMINPUTS cube.num_binary_vars
#define NUMOUTPUTS cube.part_size[cube.num_vars - 1]

#define POSITIVE_PHASE(pos)                                                    \
  (is_in_set(PLA->phase, cube.first_part[cube.output] + pos) != 0)

#define INLABEL(var) PLA->label[cube.first_part[var] + 1]
#define OUTLABEL(pos) PLA->label[cube.first_part[cube.output] + pos]

#define GETINPUT(c, pos) ((c[WHICH_WORD(2 * pos)] >> WHICH_BIT(2 * pos)) & 3)
#define GETOUTPUT(c, pos)                                                      \
  (is_in_set(c, cube.first_part[cube.output] + pos) != 0)

#define PUTINPUT(c, pos, value)                                                \
  c[WHICH_WORD(2 * pos)] =                                                     \
      (c[WHICH_WORD(2 * pos)] & ~(3 << WHICH_BIT(2 * pos))) |                  \
      (value << WHICH_BIT(2 * pos))
#define PUTOUTPUT(c, pos, value)                                               \
  c[WHICH_WORD(pos)] =                                                         \
      (c[WHICH_WORD(pos)] & (1 << WHICH_BIT(pos))) | (value << WHICH_BIT(pos))

#define TWO 3
#define DASH 3
#define ONE 2
#define ZERO 1

#define EXEC(fct, name, S)                                                     \
  {                                                                            \
    long t = ptime();                                                          \
    fct;                                                                       \
    if (trace)                                                                 \
      print_trace(S, name, ptime() - t);                                       \
  }
#define EXEC_S(fct, name, S)                                                   \
  {                                                                            \
    long t = ptime();                                                          \
    fct;                                                                       \
    if (summary)                                                               \
      print_trace(S, name, ptime() - t);                                       \
  }
#define EXECUTE(fct, i, S, cost)                                               \
  {                                                                            \
    long t = ptime();                                                          \
    fct;                                                                       \
    totals(t, i, S, &(cost));                                                  \
  }
/* lightweight EXECUTE */
#define S_EXECUTE(fct, i)                                                      \
  {                                                                            \
    long t = ptime();                                                          \
    fct;                                                                       \
    s_totals(t, i);                                                            \
  }

/*
 *    Global Variable Declarations
 */

extern unsigned int debug;                 /* debug parameter */
extern Bool verbose_debug;                 /* -v:  whether to print a lot */
extern const char *total_name[TIME_COUNT]; /* basic function names */
extern long total_time[TIME_COUNT];        /* time spent in basic fcts */
extern int total_calls[TIME_COUNT];        /* # calls to each fct */

extern Bool echo_comments;         /* turned off by -eat option */
extern Bool echo_unknown_commands; /* always true ?? */
extern Bool force_irredundant;     /* -nirr command line option */
extern Bool skip_make_sparse;
extern Bool kiss;                     /* -kiss command line option */
extern Bool pos;                      /* -pos command line option */
extern Bool print_solution;           /* -x command line option */
extern Bool recompute_onset;          /* -onset command line option */
extern Bool remove_essential;         /* -ness command line option */
extern Bool single_expand;            /* -fast command line option */
extern Bool summary;                  /* -s command line option */
extern Bool trace;                    /* -t command line option */
extern Bool unwrap_onset;             /* -nunwrap command line option */
extern Bool use_random_order;         /* -random command line option */
extern Bool use_super_gasp;           /* -strong command line option */
extern char *filename;                /* filename PLA was read from */
extern Bool debug_exact_minimization; /* dumps info for -do exact */

/*
 *  pla_types are the input and output types for reading/writing a PLA
 */
struct pla_types_struct {
  const char *key;
  int value;
};

/*
 *  The cube structure is a global structure which contains information
 *  on how a set maps into a cube -- i.e., number of parts per variable,
 *  number of variables, etc.  Also, many fields are pre-computed to
 *  speed up various primitive operations.
 */
#define CUBE_TEMP 10

struct cube_struct {
  int size;            /* set size of a cube */
  int num_vars;        /* number of variables in a cube */
  int num_binary_vars; /* number of binary variables */
  int *first_part;     /* first element of each variable */
  int *last_part;      /* first element of each variable */
  int *part_size;      /* number of elements in each variable */
  int *first_word;     /* first word for each variable */
  int *last_word;      /* last word for each variable */
  pset binary_mask;    /* Mask to extract binary variables */
  pset mv_mask;        /* mask to get mv parts */
  pset *var_mask;      /* mask to extract a variable */
  pset *temp;          /* an array of temporary sets */
  pset fullset;        /* a full cube */
  pset emptyset;       /* an empty cube */
  unsigned int inmask; /* mask to get odd word of binary part */
  int inword;          /* which word number for above */
  int *sparse;         /* should this variable be sparse? */
  int num_mv_vars;     /* number of multiple-valued variables */
  int output;          /* which variable is "output" (-1 if none) */
};

struct cdata_struct {
  int *part_zeros;   /* count of zeros for each element */
  int *var_zeros;    /* count of zeros for each variable */
  int *parts_active; /* number of "active" parts for each var */
  Bool *is_unate;    /* indicates given var is unate */
  int vars_active;   /* number of "active" variables */
  int vars_unate;    /* number of unate variables */
  int best;          /* best "binate" variable */
};

// Define the StringStream
typedef struct {
  char *s; // Pointer to the string
  int i;   // Current position in the string
} StringStream;

extern struct pla_types_struct pla_types[];
extern struct cube_struct cube, temp_cube_save;
extern struct cdata_struct cdata, temp_cdata_save;

#if BPI == 32
#define DISJOINT 0x55555555
#else
#define DISJOINT 0x5555
#endif

/* function declarations */

/* cofactor.c */ extern int binate_split_select(pset *T, pset cleft,
                                                pset cright, int debug_flag);
/* cofactor.c */ extern pset_family cubeunlist(pset *A1);
/* cofactor.c */ extern pset *cofactor(pset *T, pset c);
/* cofactor.c */ extern pset *cube1list(pset_family A);
/* cofactor.c */ extern pset *cube2list(pset_family A, pset_family B);
/* cofactor.c */ extern pset *cube3list(pset_family A, pset_family B,
                                        pset_family C);
/* cofactor.c */ extern pset *scofactor(pset *T, pset c, int var);
/* cofactor.c */ extern void massive_count(pset *T);
/* compl.c */ extern pset_family complement(pset *T);
/* compl.c */ extern pset_family simplify(pset *T);
/* compl.c */ extern void simp_comp(pset *T, pset_family *Tnew,
                                    pset_family *Tbar);
/* contain.c */ extern int d1_rm_equal(pset *A1, qsort_compare_func compare);
/* contain.c */ extern int rm2_contain(pset *A1, pset *B1);
/* contain.c */ extern int rm2_equal(pset *A1, pset *B1, pset *E1,
                                     qsort_compare_func compare);
/* contain.c */ extern int rm_contain(pset *A1);
/* contain.c */ extern int rm_equal(pset *A1, qsort_compare_func compare);
/* contain.c */ extern int rm_rev_contain(pset *A1);
/* contain.c */ extern pset *sf_list(pset_family A);
/* contain.c */ extern pset *sf_sort(pset_family A, qsort_compare_func compare);
/* contain.c */ extern pset_family d1merge(pset_family A, int var);
/* contain.c */ extern pset_family dist_merge(pset_family A, pset mask);
/* contain.c */ extern pset_family sf_contain(pset_family A);
/* contain.c */ extern pset_family sf_dupl(pset_family A);
/* contain.c */ extern pset_family sf_ind_contain(pset_family A,
                                                  int *row_indices);
/* contain.c */ extern pset_family sf_ind_unlist(pset *A1, int totcnt, int size,
                                                 int *row_indices, pset pfirst);
/* contain.c */ extern pset_family sf_merge(pset *A1, pset *B1, pset *E1,
                                            int totcnt, int size);
/* contain.c */ extern pset_family sf_rev_contain(pset_family A);
/* contain.c */ extern pset_family sf_union(pset_family A, pset_family B);
/* contain.c */ extern pset_family sf_unlist(pset *A1, int totcnt, int size);
/* cubestr.c */ extern void cube_setup(void);
/* cubestr.c */ extern void restore_cube_struct(void);
/* cubestr.c */ extern void save_cube_struct(void);
/* cubestr.c */ extern void setdown_cube(void);
/* cvrin.c */ extern void PLA_labels(pPLA PLA);
/* cvrin.c */ extern char *get_word(FILE *fp, char *word);
/* cvrin.c */ extern int label_index(pPLA PLA, char *word, int *varp, int *ip);
/* cvrin.c */ extern int read_pla(char *fp, int needs_dcset, int needs_offset,
                                  int pla_type, pPLA *PLA_return);
/* cvrin.c */ extern int read_symbolic(StringStream *fp, pPLA PLA, char *word,
                                       symbolic_t **retval);
/* cvrin.c */ extern pPLA new_PLA(void);
/* cvrin.c */ extern void PLA_summary(pPLA PLA);
/* cvrin.c */ extern void free_PLA(pPLA PLA);
/* cvrin.c */ extern void parse_pla(StringStream *fp, pPLA PLA);
/* cvrin.c */ extern void read_cube(StringStream *fp, pPLA PLA);
/* cvrin.c */ extern void skip_line(FILE *fpin, FILE *fpout, int echo);
/* cvrm.c */ extern void foreach_output_function(pPLA PLA,
                                                 int (*func)(pPLA, int),
                                                 int (*func1)(pPLA, int));
/* cvrm.c */ extern int cubelist_partition(pset *T, pset **A, pset **B,
                                           unsigned int comp_debug);
/* cvrm.c */ extern int so_both_do_espresso(pPLA PLA, int i);
/* cvrm.c */ extern int so_both_do_exact(pPLA PLA, int i);
/* cvrm.c */ extern int so_both_save(pPLA PLA, int i);
/* cvrm.c */ extern int so_do_espresso(pPLA PLA, int i);
/* cvrm.c */ extern int so_do_exact(pPLA PLA, int i);
/* cvrm.c */ extern int so_save(pPLA PLA, int i);
/* cvrm.c */ extern pset_family cof_output(pset_family T, int i);
/* cvrm.c */ extern pset_family lex_sort(pset_family T);
/* cvrm.c */ extern pset_family mini_sort(pset_family F,
                                          qsort_compare_func compare);
/* cvrm.c */ extern pset_family random_order(pset_family F);
/* cvrm.c */ extern pset_family size_sort(pset_family T);
/* cvrm.c */ extern pset_family sort_reduce(pset_family T);
/* cvrm.c */ extern pset_family uncof_output(pset_family T, int i);
/* cvrm.c */ extern pset_family unravel(pset_family B, int start);
/* cvrm.c */ extern pset_family unravel_range(pset_family B, int start,
                                              int end);
/* cvrm.c */ extern void so_both_espresso(pPLA PLA, int strategy);
/* cvrm.c */ extern void so_espresso(pPLA PLA, int strategy);
/* cvrmisc.c */ extern char *fmt_cost(pcost cost);
/* cvrmisc.c */ extern char *print_cost(pset_family F);
/* cvrmisc.c */ extern void copy_cost(pcost s, pcost d);
/* cvrmisc.c */ extern void cover_cost(pset_family F, pcost cost);
/* cvrmisc.c */ extern void fatal(const char *s);
/* cvrmisc.c */ extern void print_trace(pset_family T, const char *name,
                                        long time);
/* cvrmisc.c */ extern void size_stamp(pset_family T, const char *name);
/* cvrmisc.c */ extern void totals(long time, int i, pset_family T, pcost cost);
/* cvrout.c */ extern char *fmt_cube(pset c, const char *out_map, char *s);
/* cvrout.c */ extern char *pc1(pset c);
/* cvrout.c */ extern char *pc2(pset c);
/* cvrout.c */ extern void makeup_labels(pPLA PLA);
/* cvrout.c */ extern void kiss_output(FILE *fp, pPLA PLA);
/* cvrout.c */ extern void kiss_print_cube(FILE *fp, pPLA PLA, pset p,
                                           const char *out_string);
/* cvrout.c */ extern void output_symbolic_constraints(FILE *fp, pPLA PLA,
                                                       int output_symbolic);
/* cvrout.c */ extern void cprint(pset_family T);
/* cvrout.c */ extern void debug1_print(pset_family T, const char *name,
                                        int num);
/* cvrout.c */ extern void debug_print(pset *T, const char *name, int level);
/* cvrout.c */ extern char *eqn_output(pPLA PLA);
/* cvrout.c */ extern void fpr_header(FILE *fp, pPLA PLA, int output_type);
/* cvrout.c */ extern void pls_group(pPLA PLA, FILE *fp);
/* cvrout.c */ extern void pls_label(pPLA PLA, FILE *fp);
/* cvrout.c */ extern void pls_output(pPLA PLA);
/* cvrout.c */ extern void print_cube(FILE *fp, pset c, const char *out_map);
/* cvrout.c */ extern void print_expanded_cube(FILE *fp, pset c, pset phase);
/* equiv.c */ extern void find_equiv_outputs(pPLA PLA);
/* equiv.c */ extern int check_equiv(pset_family f1, pset_family f2);
/* espresso.c */ extern pset_family espresso(pset_family F, pset_family D1,
                                             pset_family R);
/* essen.c */ extern int essen_cube(pset_family F, pset_family D, pset c);
/* essen.c */ extern pset_family cb_consensus(pset_family T, pset c);
/* essen.c */ extern pset_family cb_consensus_dist0(pset_family R, pset p,
                                                    pset c);
/* essen.c */ extern pset_family essential(pset_family *Fp, pset_family *Dp);
/* exact.c */ extern pset_family minimize_exact(pset_family F, pset_family D,
                                                pset_family R, int exact_cover);
/* exact.c */ extern pset_family minimize_exact_literals(pset_family F,
                                                         pset_family D,
                                                         pset_family R,
                                                         int exact_cover);
/* expand.c */ extern int feasibly_covered(pset_family BB, pset c, pset RAISE,
                                           pset new_lower);
/* expand.c */ extern int most_frequent(pset_family CC, pset FREESET);
/* expand.c */ extern pset_family all_primes(pset_family F, pset_family R);
/* expand.c */ extern pset_family expand(pset_family F, pset_family R,
                                         int nonsparse);
/* expand.c */ extern pset_family find_all_primes(pset_family BB, pset RAISE,
                                                  pset FREESET);
/* expand.c */ extern void elim_lowering(pset_family BB, pset_family CC,
                                         pset RAISE, pset FREESET);
/* expand.c */ extern void essen_parts(pset_family BB, pset_family CC,
                                       pset RAISE, pset FREESET);
/* expand.c */ extern void essen_raising(pset_family BB, pset RAISE,
                                         pset FREESET);
/* expand.c */ extern void expand1(pset_family BB, pset_family CC, pset RAISE,
                                   pset FREESET, pset OVEREXPANDED_CUBE,
                                   pset SUPER_CUBE, pset INIT_LOWER,
                                   int *num_covered, pset c);
/* expand.c */ extern void mincov(pset_family BB, pset RAISE, pset FREESET);
/* expand.c */ extern void select_feasible(pset_family BB, pset_family CC,
                                           pset RAISE, pset FREESET,
                                           pset SUPER_CUBE, int *num_covered);
/* expand.c */ extern void setup_BB_CC(pset_family BB, pset_family CC);
/* gasp.c */ extern pset_family
expand_gasp(pset_family F, pset_family D, pset_family R, pset_family Foriginal);
/* gasp.c */ extern pset_family irred_gasp(pset_family F, pset_family D,
                                           pset_family G);
/* gasp.c */ extern pset_family last_gasp(pset_family F, pset_family D,
                                          pset_family R, cost_t *cost);
/* gasp.c */ extern pset_family super_gasp(pset_family F, pset_family D,
                                           pset_family R, cost_t *cost);
/* gasp.c */ extern void expand1_gasp(pset_family F, pset_family D,
                                      pset_family R, pset_family Foriginal,
                                      int c1index, pset_family *G);
/* hack.c */ extern void find_inputs(pset_family A, pPLA PLA,
                                     symbolic_list_t *list, int base, int value,
                                     pset_family *newF, pset_family *newD);
/* hack.c */ extern void form_bitvector(pset p, int base, int value,
                                        symbolic_list_t *list);
/* hack.c */ extern void map_dcset(pPLA PLA);
/* hack.c */ extern void map_output_symbolic(pPLA PLA);
/* hack.c */ extern void map_symbolic(pPLA PLA);
/* hack.c */ extern pset_family
map_symbolic_cover(pset_family T, symbolic_list_t *list, int base);
/* hack.c */ extern void symbolic_hack_labels(pPLA PLA, symbolic_t *list,
                                              pset compress, int new_size,
                                              int old_size, int size_added);
/* hack.c */ extern void disassemble_fsm(pPLA PLA, int verbose_mode);
/* irred.c */ extern int cube_is_covered(pset *T, pset c);
/* irred.c */ extern int taut_special_cases(pset *T);
/* irred.c */ extern int tautology(pset *T);
/* irred.c */ extern pset_family irredundant(pset_family F, pset_family D);
/* irred.c */ extern void mark_irredundant(pset_family F, pset_family D);
/* irred.c */ extern void irred_split_cover(pset_family F, pset_family D,
                                            pset_family *E, pset_family *Rt,
                                            pset_family *Rp);
/* irred.c */ extern sm_matrix *irred_derive_table(pset_family D, pset_family E,
                                                   pset_family Rp);
/* map.c */ extern pset minterms(pset_family T);
/* map.c */ extern void explode(int var, int z);
/* map.c */ extern void map(pset_family T);
/* opo.c */ extern void output_phase_setup(pPLA PLA, int first_output);
/* opo.c */ extern pPLA set_phase(pPLA PLA);
/* opo.c */ extern pset_family opo(pset phase, pset_family T, pset_family D,
                                   pset_family R, int first_output);
/* opo.c */ extern pset find_phase(pPLA PLA, int first_output, pset phase1);
/* opo.c */ extern pset_family find_covers(pcover F, pcover D, pset select,
                                           int n);
/* opo.c */ extern pset_family form_cover_table(pcover F, pcover D, pset select,
                                                int f, int fbar);
/* opo.c */ extern pset_family opo_leaf(pset_family T, pset select, int out1,
                                        int out2);
/* opo.c */ extern pset_family opo_recur(pset_family T, pset_family D,
                                         pset select, int offset, int first,
                                         int last);
/* opo.c */ extern void opoall(pPLA PLA, int first_output, int last_output,
                               int opo_strategy);
/* opo.c */ extern void phase_assignment(pPLA PLA, int opo_strategy);
/* opo.c */ extern void repeated_phase_assignment(pPLA PLA);
/* pair.c */ extern void generate_all_pairs(ppair pair, int n, pset candidate,
                                            void (*action)(ppair));
/* pair.c */ extern void find_best_cost(ppair pair);
/* pair.c */ extern int greedy_best_cost(int **cost_array_local, ppair *pair_p);
/* pair.c */ extern void minimize_pair(ppair pair);
/* pair.c */ extern void pair_free(ppair pair);
/* pair.c */ extern void pair_all(pPLA PLA, int pair_strategy);
/* pair.c */ extern pset_family delvar(pset_family A, int paired[]);
/* pair.c */ extern pset_family pairvar(pset_family A, ppair pair);
/* pair.c */ extern ppair pair_best_cost(int **cost_array_local);
/* pair.c */ extern ppair pair_new(int n);
/* pair.c */ extern ppair pair_save(ppair pair, int n);
/* pair.c */ extern void print_pair(ppair pair);
/* pair.c */ extern void find_optimal_pairing(pPLA PLA, int strategy);
/* pair.c */ extern void set_pair(pPLA PLA);
/* pair.c */ extern void set_pair1(pPLA PLA, int adjust_labels);
/* primes.c */ extern pset_family primes_consensus(pset *T);
/* reduce.c */ extern int sccc_special_cases(pset *T, pset *result);
/* reduce.c */ extern pset_family reduce(pset_family F, pset_family D);
/* reduce.c */ extern pset reduce_cube(pset *FD, pset p);
/* reduce.c */ extern pset sccc(pset *T);
/* reduce.c */ extern pset sccc_cube(pset result, pset p);
/* reduce.c */ extern pset sccc_merge(pset left, pset right, pset cl, pset cr);
/* set.c */ extern int set_andp(pset r, pset a, pset b);
/* set.c */ extern int set_orp(pset r, pset a, pset b);
/* set.c */ extern int setp_disjoint(pset a, pset b);
/* set.c */ extern int setp_empty(pset a);
/* set.c */ extern int setp_equal(pset a, pset b);
/* set.c */ extern int setp_full(pset a, int size);
/* set.c */ extern int setp_implies(pset a, pset b);
/* set.c */ extern char *pbv1(pset s, int n);
/* set.c */ extern char *ps1(pset a);
/* set.c */ extern int *sf_count(pset_family A);
/* set.c */ extern int *sf_count_restricted(pset_family A, pset r);
/* set.c */ extern int bit_index(unsigned int a);
/* set.c */ extern int set_dist(pset a, pset b);
/* set.c */ extern int set_ord(pset a);
/* set.c */ extern void set_adjcnt(pset a, int *count, int weight);
/* set.c */ extern pset set_and(pset r, pset a, pset b);
/* set.c */ extern pset set_clear(pset r, int size);
/* set.c */ extern pset set_copy(pset r, pset a);
/* set.c */ extern pset set_diff(pset r, pset a, pset b);
/* set.c */ extern pset set_fill(pset r, int size);
/* set.c */ extern pset set_merge(pset r, pset a, pset b, pset mask);
/* set.c */ extern pset set_or(pset r, pset a, pset b);
/* set.c */ extern pset set_xor(pset r, pset a, pset b);
/* set.c */ extern pset sf_and(pset_family A);
/* set.c */ extern pset sf_or(pset_family A);
/* set.c */ extern pset_family sf_active(pset_family A);
/* set.c */ extern pset_family sf_addcol(pset_family A, int firstcol, int n);
/* set.c */ extern pset_family sf_addset(pset_family A, pset s);
/* set.c */ extern pset_family sf_append(pset_family A, pset_family B);
/* set.c */ extern pset_family sf_bm_read(FILE *fp);
/* set.c */ extern pset_family sf_compress(pset_family A, pset c);
/* set.c */ extern pset_family sf_copy(pset_family R, pset_family A);
/* set.c */ extern pset_family sf_copy_col(pset_family dst, int dstcol,
                                           pset_family src, int srccol);
/* set.c */ extern pset_family sf_delc(pset_family A, int first, int last);
/* set.c */ extern pset_family sf_delcol(pset_family A, int firstcol, int n);
/* set.c */ extern pset_family sf_inactive(pset_family A);
/* set.c */ extern pset_family sf_join(pset_family A, pset_family B);
/* set.c */ extern pset_family sf_new(int num, int size);
/* set.c */ extern pset_family sf_permute(pset_family A, int *permute,
                                          int npermute);
/* set.c */ extern pset_family sf_read(FILE *fp);
/* set.c */ extern pset_family sf_save(pset_family A);
/* set.c */ extern pset_family sf_transpose(pset_family A);
/* set.c */ extern void set_write(FILE *fp, pset a);
/* set.c */ extern void sf_bm_print(pset_family A);
/* set.c */ extern void sf_cleanup(void);
/* set.c */ extern void sf_delset(pset_family A, int i);
/* set.c */ extern void sf_free(pset_family A);
/* set.c */ extern void sf_print(pset_family A);
/* set.c */ extern void sf_write(FILE *fp, pset_family A);
/* setc.c */ extern int ccommon(pset a, pset b, pset cof);
/* setc.c */ extern int cdist0(pset a, pset b);
/* setc.c */ extern int full_row(pset p, pset cof);
/* setc.c */ extern int ascend(pset *a, pset *b);
/* setc.c */ extern int cactive(pset a);
/* setc.c */ extern int cdist(pset a, pset b);
/* setc.c */ extern int cdist01(pset a, pset b);
/* setc.c */ extern int d1_order(pset *a, pset *b);
/* setc.c */ extern int desc1(pset a, pset b);
/* setc.c */ extern int descend(pset *a, pset *b);
/* setc.c */ extern int lex_order(pset *a, pset *b);
/* setc.c */ extern pset force_lower(pset xlower, pset a, pset b);
/* setc.c */ extern void consensus(pset r, pset a, pset b);
/* sharp.c */ extern pset_family cb1_dsharp(pset_family T, pset c);
/* sharp.c */ extern pset_family cb_dsharp(pset c, pset_family T);
/* sharp.c */ extern pset_family cb_recur_sharp(pset c, pset_family T,
                                                int first, int last, int level);
/* sharp.c */ extern pset_family cb_sharp(pset c, pset_family T);
/* sharp.c */ extern pset_family cv_dsharp(pset_family A, pset_family B);
/* sharp.c */ extern pset_family cv_intersect(pset_family A, pset_family B);
/* sharp.c */ extern pset_family cv_sharp(pset_family A, pset_family B);
/* sharp.c */ extern pset_family dsharp(pset a, pset b);
/* sharp.c */ extern pset_family make_disjoint(pset_family A);
/* sharp.c */ extern pset_family sharp(pset a, pset b);
/* signature.c */ extern pset_family signature(pset_family F1, pset_family D1,
                                               pset_family R1);
/* sminterf.c */ pset do_sm_minimum_cover(pset_family A);
/* sparse.c */ extern pset_family make_sparse(pset_family F, pset_family D,
                                              pset_family R);
/* sparse.c */ extern pset_family mv_reduce(pset_family F, pset_family D);
/* ucbqsort.c AB */ /* extern qsort(); */
/* unate.c */ extern pset_family map_cover_to_unate(pset *T);
/* unate.c */ extern pset_family map_unate_to_cover(pset_family A);
/* unate.c */ extern pset_family exact_minimum_cover(pset_family T);
/* unate.c */ extern pset_family unate_compl(pset_family A);
/* unate.c */ extern pset_family unate_complement(pset_family A);
/* unate.c */ extern pset_family unate_intersect(pset_family A, pset_family B,
                                                 int largest_only);
/* verify.c */ extern void PLA_permute(pPLA PLA1, pPLA PLA2);
/* verify.c */ extern int PLA_verify(pPLA PLA1, pPLA PLA2);
/* verify.c */ extern int check_consistency(pPLA PLA);
/* verify.c */ extern int verify(pset_family F, pset_family Fold,
                                 pset_family Dold);

#endif // DYNAMATIC_SUPPORT_ESPRESSO_ESPRESSO_H