#include "dynamatic/Support/Espresso/espresso.h"
#include "dynamatic/Support/Espresso/signature.h"
#include <stdio.h>

void setup_bw(pset_family R, pset c);
int black_white(void);
void split_list(pset_family R, int v);
void merge_list(void);

static void alloc_list(int size);
static void free_list(void);
static void init_list(int size);
static void delete_elem(int element);
static void insert(int element);
static void print_links(int size, int *list);
void print_bw(int size);

static void alloc_stack(int size);
static void free_stack(void);
static void clear(void);

static int white_head, white_tail;
static int black_head, black_tail;
static int forward_link, backward_link;
static int *forward, *backward;

static int *stack_head, *stack_tail, stack_p;

static pcover BB;

static void alloc_list(int size) {
  forward = (int *)calloc(size, sizeof(int));
  backward = (int *)calloc(size, sizeof(int));
  if (!forward || !backward) {
    perror("alloc_list");
    exit(1);
  }
}

static void free_list(void) {
  free(forward);
  free(backward);
}

static void init_list(int size) {
  int i;
  for (i = 0; i < size; i++) {
    forward[i] = i + 1;
    backward[i] = i - 1;
  }
  forward[size - 1] = -1;
  backward[0] = -1;
  white_head = 0;
  white_tail = size - 1;
  black_head = -1;
  black_tail = -1;
}

static void delete_elem(int element) {
  forward_link = forward[element];
  backward_link = backward[element];
  if (forward_link != -1) {
    if (backward_link != -1) {
      forward[backward_link] = forward_link;
      backward[forward_link] = backward_link;
    } else {
      white_head = forward_link;
      backward[forward_link] = -1;
    }
  } else {
    if (backward_link != -1) {
      white_tail = backward_link;
      forward[backward_link] = -1;
    } else {
      white_head = white_tail = -1;
    }
  }
}

static void insert(int element) {
  if (black_head != -1) {
    forward[element] = black_head;
    backward[element] = -1;
    backward[black_head] = element;
    black_head = element;
  } else {
    black_head = black_tail = element;
    forward[element] = backward[element] = -1;
  }
}

void merge_list(void) {
  if (white_head != -1) {
    if (black_head != -1) {
      forward[white_tail] = black_head;
      backward[black_head] = white_tail;
      white_tail = black_tail;
      black_head = black_tail = -1;
    }
  } else {
    white_head = black_head;
    white_tail = black_tail;
    black_head = black_tail = -1;
  }
}

static void print_links(int size, int *list) {
  int i;
  for (i = 0; i < size; i++) {
    printf("%d%c", list[i], (i + 1) % 10 ? '\t' : '\n');
  }
  printf("\n");
}

void print_bw(int size) {
  printf("white_head %d\twhite_tail %d\tblack_head %d\tblack_tail %d\n",
         white_head, white_tail, black_head, black_tail);
  print_links(size, forward);
  print_links(size, backward);
}

static void alloc_stack(int size) {
  stack_head = (int *)calloc(size, sizeof(int));
  stack_tail = (int *)calloc(size, sizeof(int));
  if (!stack_head || !stack_tail) {
    perror("alloc_stack");
    exit(1);
  }
}

static void free_stack(void) {
  free(stack_head);
  free(stack_tail);
}

void push_black_list(void) {
  stack_head[stack_p] = black_head;
  stack_tail[stack_p++] = black_tail;
}

void pop_black_list(void) {
  black_head = stack_head[--stack_p];
  black_tail = stack_tail[stack_p];
}

static void clear(void) { stack_p = 0; }

void setup_bw(pset_family R, pset c) {
  pcube out_part_r;
  int size = R->count;
  int i;
  pcube b, r;
  int w, last;
  unsigned int x;

  /* Allocate memory */
  alloc_list(size);
  alloc_stack(cube.num_binary_vars);
  BB = new_cover(size);
  BB->count = size;
  out_part_r = new_cube();

  init_list(size);
  clear();
  /* form BB */
  /* Blocking Matrix formed here is bit-complement of that in sigma  */
  foreachi_set(R, i, r) {
    b = GETSET(BB, i);
    if ((last = cube.inword) != -1) {
      /* Check the partial word of binary variables */
      x = r[last] & c[last];
      x = ~(x | x >> 1) & cube.inmask;
      b[last] = r[last] & (x | x << 1);
      /* Check the full words of binary variables */
      for (w = 1; w < last; w++) {
        x = r[w] & c[w];
        x = ~(x | x >> 1) & DISJOINT;
        b[w] = r[w] & (x | x << 1);
      }
    }
    PUTLOOP(b, LOOP(r));
    INLINEset_and(b, b, cube.binary_mask);
    INLINEset_and(out_part_r, cube.mv_mask, r);
    if (!setp_implies(out_part_r, c)) {
      INLINEset_or(b, b, out_part_r);
    }
    set_not(b);
  }
  free_cube(out_part_r);
}

void free_bw(void) {
  free_list();
  free_stack();
  free_cover(BB);
}

int black_white(void) {
  int b_index, w_index;
  int containment;
  for (b_index = black_head; b_index != -1; b_index = forward[b_index]) {
    containment = FALSE;
    for (w_index = white_head; w_index != -1; w_index = forward[w_index]) {
      if (setp_implies(GETSET(BB, b_index), GETSET(BB, w_index))) {
        containment = TRUE;
        break;
      }
    }
    if (containment == FALSE) {
      return FALSE;
    }
  }
  return TRUE;
}

void reset_black_list(void) { black_head = black_tail = -1; }

void split_list(pset_family R, int v) {
  int index, next_index;
  ;

  for (index = white_head; index != -1; index = next_index) {
    next_index = forward[index];
    if (!is_in_set(GETSET(R, index), v)) {
      delete_elem(index);
      insert(index);
    }
  }
}

/* Data Structures for ordering variables in ess_test_and_reduction */
static int variable_count; /* Number of variables currently in the list */
static int *variable_forward_chain;  /* Next */
static int *variable_backward_chain; /* Previous */
static int variable_head;            /* first element in the list */
static int variable_tail;            /* last element in the list */

void variable_list_alloc(int size) {
  variable_forward_chain = (int *)calloc(size, sizeof(int));
  variable_backward_chain = (int *)calloc(size, sizeof(int));
  if (!variable_forward_chain || !variable_backward_chain) {
    perror("variable_list_alloc");
    exit(1);
  }
}

void variable_list_init(int reduced_c_free_count, int *reduced_c_free_list) {
  int i;
  int v, next_v;

  variable_count = reduced_c_free_count;
  if (variable_count == 0) {
    variable_head = variable_tail = -1;
    return;
  }
  variable_head = reduced_c_free_list[0];
  variable_tail = reduced_c_free_list[variable_count - 1];
  variable_forward_chain[variable_tail] = -1;
  variable_backward_chain[variable_head] = -1;

  next_v = variable_head;
  for (i = 1; i < variable_count; i++) {
    v = next_v;
    next_v = reduced_c_free_list[i];
    variable_forward_chain[v] = next_v;
    variable_backward_chain[next_v] = v;
  }
}

void variable_list_delete(int element) {
  variable_count--;
  forward_link = variable_forward_chain[element];
  backward_link = variable_backward_chain[element];
  if (forward_link != -1) {
    if (backward_link != -1) {
      variable_forward_chain[backward_link] = forward_link;
      variable_backward_chain[forward_link] = backward_link;
    } else {
      variable_head = forward_link;
      variable_backward_chain[forward_link] = -1;
    }
  } else {
    if (backward_link != -1) {
      variable_tail = backward_link;
      variable_forward_chain[backward_link] = -1;
    } else {
      variable_head = variable_tail = -1;
    }
  }
}

void variable_list_insert(int element) {
  variable_count++;
  if (variable_head != -1) {
    variable_forward_chain[element] = variable_head;
    variable_backward_chain[element] = -1;
    variable_backward_chain[variable_head] = element;
    variable_head = element;
  } else {
    variable_head = variable_tail = element;
    variable_forward_chain[element] = -1;
    variable_backward_chain[element] = -1;
  }
}

int variable_list_empty(void) { return variable_count ? FALSE : TRUE; }

void get_next_variable(int *pv, int *pphase, pset_family R) {
  int v, e0, e1;
  int e0_black_count, e1_black_count;
  int w_index;
  int max_black_count, max_variable = 0, max_phase = 0;
  pcube r;

  max_black_count = -1;
  for (v = variable_head; v != -1; v = variable_forward_chain[v]) {
    e0 = v << 1;
    e1 = e0 + 1;
    e0_black_count = e1_black_count = 0;
    for (w_index = white_head; w_index != -1; w_index = forward[w_index]) {
      r = GETSET(R, w_index);
      if (is_in_set(r, e0)) {
        if (is_in_set(r, e1)) {
          continue;
        } else {
          e1_black_count++;
        }
      } else {
        e0_black_count++;
      }
    }
    if (e0_black_count > e1_black_count) {
      if (e0_black_count > max_black_count) {
        max_black_count = e0_black_count;
        max_variable = v;
        max_phase = 0;
      }
    } else {
      if (e1_black_count > max_black_count) {
        max_black_count = e1_black_count;
        max_variable = v;
        max_phase = 1;
      }
    }
  }
  *pv = max_variable;
  *pphase = max_phase;
}

void print_variable_list(void) {
  printf("Variable_Forward_Chain:\n");
  print_links(cube.num_binary_vars, variable_forward_chain);
  printf("Variable_Backward_Chain:\n");
  print_links(cube.num_binary_vars, variable_backward_chain);
}
