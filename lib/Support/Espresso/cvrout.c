/*
    module: cvrout.c
    purpose: cube and cover output routines
*/

#include "dynamatic/Support/Espresso/espresso.h"

void fpr_header(FILE *fp, pPLA PLA, int output_type) {
  int i, var;
  int first, last;

  /* .type keyword gives logical type */
  if (output_type != F_type) {
    fprintf(fp, ".type ");
    if (output_type & F_type)
      putc('f', fp);
    if (output_type & D_type)
      putc('d', fp);
    if (output_type & R_type)
      putc('r', fp);
    putc('\n', fp);
  }

  /* Check for binary or multiple-valued labels */
  if (cube.num_mv_vars <= 1) {
    fprintf(fp, ".i %d\n", cube.num_binary_vars);
    if (cube.output != -1)
      fprintf(fp, ".o %d\n", cube.part_size[cube.output]);
  } else {
    fprintf(fp, ".mv %d %d", cube.num_vars, cube.num_binary_vars);
    for (var = cube.num_binary_vars; var < cube.num_vars; var++)
      fprintf(fp, " %d", cube.part_size[var]);
    fprintf(fp, "\n");
  }

  /* binary valued labels */
  if (PLA->label != NIL(char *) && PLA->label[1] != NIL(char) &&
      cube.num_binary_vars > 0) {
    fprintf(fp, ".ilb");
    for (var = 0; var < cube.num_binary_vars; var++)
      fprintf(fp, " %s", INLABEL(var));
    putc('\n', fp);
  }

  /* output-part (last multiple-valued variable) labels */
  if (PLA->label != NIL(char *) &&
      PLA->label[cube.first_part[cube.output]] != NIL(char) &&
      cube.output != -1) {
    fprintf(fp, ".ob");
    for (i = 0; i < cube.part_size[cube.output]; i++)
      fprintf(fp, " %s", OUTLABEL(i));
    putc('\n', fp);
  }

  /* multiple-valued labels */
  for (var = cube.num_binary_vars; var < cube.num_vars - 1; var++) {
    first = cube.first_part[var];
    last = cube.last_part[var];
    if (PLA->label != NULL && PLA->label[first] != NULL) {
      fprintf(fp, ".label var=%d", var);
      for (i = first; i <= last; i++) {
        fprintf(fp, " %s", PLA->label[i]);
      }
      putc('\n', fp);
    }
  }

  if (PLA->phase != (pcube)NULL) {
    first = cube.first_part[cube.output];
    last = cube.last_part[cube.output];
    fprintf(fp, "#.phase ");
    for (i = first; i <= last; i++)
      putc(is_in_set(PLA->phase, i) ? '1' : '0', fp);
    fprintf(fp, "\n");
  }
}

void pls_output(pPLA PLA) {
  pcube last, p;

  printf(".option unmerged\n");
  makeup_labels(PLA);
  pls_label(PLA, stdout);
  pls_group(PLA, stdout);
  printf(".p %d\n", PLA->F->count);
  foreach_set(PLA->F, last, p) { print_expanded_cube(stdout, p, PLA->phase); }
  printf(".end\n");
}

void pls_group(pPLA PLA, FILE *fp) {
  int var, i, col, len;

  fprintf(fp, "\n.group");
  col = 6;
  for (var = 0; var < cube.num_vars - 1; var++) {
    fprintf(fp, " ("), col += 2;
    for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
      len = strlen(PLA->label[i]);
      if (col + len > 75)
        fprintf(fp, " \\\n"), col = 0;
      else if (i != 0)
        putc(' ', fp), col += 1;
      fprintf(fp, "%s", PLA->label[i]), col += len;
    }
    fprintf(fp, ")"), col += 1;
  }
  fprintf(fp, "\n");
}

void pls_label(pPLA PLA, FILE *fp) {
  int var, i, col, len;

  fprintf(fp, ".label");
  col = 6;
  for (var = 0; var < cube.num_vars; var++)
    for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
      len = strlen(PLA->label[i]);
      if (col + len > 75)
        fprintf(fp, " \\\n"), col = 0;
      else
        putc(' ', fp), col += 1;
      fprintf(fp, "%s", PLA->label[i]), col += len;
    }
}

/*
    eqntott output mode -- output algebraic equations
*/
char *eqn_output(pPLA PLA) {
  pcube p, last;
  int i, var, col, len;
  int x;
  Bool firstand, firstor;

  if (cube.output == -1)
    fatal("Cannot have no-output function for EQNTOTT output mode");
  if (cube.num_mv_vars != 1)
    fatal("Must have binary-valued function for EQNTOTT output mode");
  makeup_labels(PLA);

  // Initial output size
  size_t output_size = 1024;

  char *output = (char *)malloc(output_size);
  if (output == NULL) {
    return NULL; // Memory allocation failed
  }
  output[0] = '\0';
  size_t current_length = 0; // To track the current length of the string

  /* Write a single equation for each output */
  for (i = 0; i < cube.part_size[cube.output]; i++) {
    // printf("%s = ", OUTLABEL(i));
    char temp[256];
    int temp_length = snprintf(temp, sizeof(temp), "%s = ", OUTLABEL(i));
    current_length += temp_length;
    // Check if buffer needs to be resized
    if (current_length >= output_size) {
      output_size *= 2;
      output = (char *)realloc(output, output_size);
      if (output == NULL) {
        return NULL; // Memory reallocation failed
      }
    }
    strcat(output, temp);

    col = strlen(OUTLABEL(i)) + 3;
    firstor = TRUE;

    /* Write product terms for each cube in this output */
    foreach_set(PLA->F, last,
                p) if (is_in_set(p, i + cube.first_part[cube.output])) {
      if (firstor)
        strcat(output, "("), col += 1;
      else
        strcat(output, " | ("), col += 4;
      firstor = FALSE;
      firstand = TRUE;

      // print out a product term
      for (var = 0; var < cube.num_binary_vars; var++) {
        if ((x = GETINPUT(p, var)) != DASH) {
          len = strlen(INLABEL(var));
          if (col + len > 72) {
            strcat(output, "\n    ");
            col = 4;
          }
          if (!firstand) {
            strcat(output, "&");
            col += 1;
          }
          firstand = FALSE;
          if (x == ZERO) {
            strcat(output, "!");
            col += 1;
          }
          strcat(output, INLABEL(var));
          col += len;
          current_length += len + (x == ZERO ? 1 : 0) + (firstand ? 0 : 1);
          // Check if buffer needs to be resized
          if (current_length >= output_size) {
            output_size *= 2;
            output = (char *)realloc(output, output_size);
            if (output == NULL) {
              return NULL; // Memory reallocation failed
            }
          }
        }
      }
      strcat(output, ")");
      col += 1;
    }
    strcat(output, ";\n\n");
  }
  return output;
}

char *fmt_cube(pset c, const char *out_map, char *s) {
  int i, var, last, len = 0;

  for (var = 0; var < cube.num_binary_vars; var++) {
    s[len++] = "?01-"[GETINPUT(c, var)];
  }
  for (var = cube.num_binary_vars; var < cube.num_vars - 1; var++) {
    s[len++] = ' ';
    for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
      s[len++] = "01"[is_in_set(c, i) != 0];
    }
  }
  if (cube.output != -1) {
    last = cube.last_part[cube.output];
    s[len++] = ' ';
    for (i = cube.first_part[cube.output]; i <= last; i++) {
      s[len++] = out_map[is_in_set(c, i) != 0];
    }
  }
  s[len] = '\0';
  return s;
}

void print_cube(FILE *fp, pset c, const char *out_map) {
  int i, var, ch;
  int last;

  for (var = 0; var < cube.num_binary_vars; var++) {
    ch = "?01-"[GETINPUT(c, var)];
    putc(ch, fp);
  }
  for (var = cube.num_binary_vars; var < cube.num_vars - 1; var++) {
    putc(' ', fp);
    for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
      ch = "01"[is_in_set(c, i) != 0];
      putc(ch, fp);
    }
  }
  if (cube.output != -1) {
    last = cube.last_part[cube.output];
    putc(' ', fp);
    for (i = cube.first_part[cube.output]; i <= last; i++) {
      ch = out_map[is_in_set(c, i) != 0];
      putc(ch, fp);
    }
  }
  putc('\n', fp);
}

void print_expanded_cube(FILE *fp, pset c, pset phase) {
  int i, var, ch;
  const char *out_map;

  for (var = 0; var < cube.num_binary_vars; var++) {
    for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
      ch = "~1"[is_in_set(c, i) != 0];
      putc(ch, fp);
    }
  }
  for (var = cube.num_binary_vars; var < cube.num_vars - 1; var++) {
    for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
      ch = "1~"[is_in_set(c, i) != 0];
      putc(ch, fp);
    }
  }
  if (cube.output != -1) {
    var = cube.num_vars - 1;
    putc(' ', fp);
    for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
      if (phase == (pcube)NULL || is_in_set(phase, i)) {
        out_map = "~1";
      } else {
        out_map = "~0";
      }
      ch = out_map[is_in_set(c, i) != 0];
      putc(ch, fp);
    }
  }
  putc('\n', fp);
}

char *pc1(pset c) {
  static char s1[256];
  return fmt_cube(c, "01", s1);
}
char *pc2(pset c) {
  static char s2[256];
  return fmt_cube(c, "01", s2);
}

void debug_print(pset *T, const char *name, int level) {
  pcube *T1, p, temp;
  int cnt;

  cnt = CUBELISTSIZE(T);
  temp = new_cube();
  if (verbose_debug && level == 0)
    printf("\n");
  printf("%s[%d]: ord(T)=%d\n", name, level, cnt);
  if (verbose_debug) {
    printf("cofactor=%s\n", pc1(T[0]));
    for (T1 = T + 2, cnt = 1; (p = *T1++) != (pcube)NULL; cnt++)
      printf("%4d. %s\n", cnt, pc1(set_or(temp, p, T[0])));
  }
  free_cube(temp);
}

void debug1_print(pset_family T, const char *name, int num) {
  int cnt = 1;
  pcube p, last;

  if (verbose_debug && num == 0)
    printf("\n");
  printf("%s[%d]: ord(T)=%d\n", name, num, T->count);
  if (verbose_debug)
    foreach_set(T, last, p) printf("%4d. %s\n", cnt++, pc1(p));
}

void cprint(pset_family T) {
  pcube p, last;

  foreach_set(T, last, p) printf("%s\n", pc1(p));
}

void makeup_labels(pPLA PLA) {
  int var, i, ind;

  if (PLA->label == (char **)NULL)
    PLA_labels(PLA);

  for (var = 0; var < cube.num_vars; var++)
    for (i = 0; i < cube.part_size[var]; i++) {
      ind = cube.first_part[var] + i;
      if (PLA->label[ind] == (char *)NULL) {
        PLA->label[ind] = ALLOC(char, 15);
        if (var < cube.num_binary_vars)
          if ((i % 2) == 0)
            (void)sprintf(PLA->label[ind], "v%d.bar", var);
          else
            (void)sprintf(PLA->label[ind], "v%d", var);
        else
          (void)sprintf(PLA->label[ind], "v%d.%d", var, i);
      }
    }
}

void kiss_output(FILE *fp, pPLA PLA) {
  pset last, p;

  foreach_set(PLA->F, last, p) { kiss_print_cube(fp, PLA, p, "~1"); }
  foreach_set(PLA->D, last, p) { kiss_print_cube(fp, PLA, p, "~2"); }
}

void kiss_print_cube(FILE *fp, pPLA PLA, pset p, const char *out_string) {
  int i, var;
  int part, x;

  for (var = 0; var < cube.num_binary_vars; var++) {
    x = "?01-"[GETINPUT(p, var)];
    putc(x, fp);
  }

  for (var = cube.num_binary_vars; var < cube.num_vars - 1; var++) {
    putc(' ', fp);
    if (setp_implies(cube.var_mask[var], p)) {
      putc('-', fp);
    } else {
      part = -1;
      for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
        if (is_in_set(p, i)) {
          if (part != -1) {
            fatal("more than 1 part in a symbolic variable\n");
          }
          part = i;
        }
      }
      if (part == -1) {
        putc('~', fp); /* no parts, hope its an output ... */
      } else {
        (void)fputs(PLA->label[part], fp);
      }
    }
  }

  if ((var = cube.output) != -1) {
    putc(' ', fp);
    for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
      x = out_string[is_in_set(p, i) != 0];
      putc(x, fp);
    }
  }

  putc('\n', fp);
}

void output_symbolic_constraints(FILE *fp, pPLA PLA, int output_symbolic) {
  pset_family A;
  int i, j;
  int size, var, npermute, *permute, *weight, noweight;

  if ((cube.num_vars - cube.num_binary_vars) <= 1) {
    return;
  }
  makeup_labels(PLA);

  for (var = cube.num_binary_vars; var < cube.num_vars - 1; var++) {

    /* pull out the columns for variable "var" */
    npermute = cube.part_size[var];
    permute = ALLOC(int, npermute);
    for (i = 0; i < npermute; i++) {
      permute[i] = cube.first_part[var] + i;
    }
    A = sf_permute(sf_save(PLA->F), permute, npermute);
    FREE(permute);

    /* Delete the singletons and the full sets */
    noweight = 0;
    for (i = 0; i < A->count; i++) {
      size = set_ord(GETSET(A, i));
      if (size == 1 || size == A->sf_size) {
        sf_delset(A, i--);
        noweight++;
      }
    }

    /* Count how many times each is duplicated */
    weight = ALLOC(int, A->count);
    for (i = 0; i < A->count; i++) {
      RESET(GETSET(A, i), COVERED);
    }
    for (i = 0; i < A->count; i++) {
      weight[i] = 0;
      if (!TESTP(GETSET(A, i), COVERED)) {
        weight[i] = 1;
        for (j = i + 1; j < A->count; j++) {
          if (setp_equal(GETSET(A, i), GETSET(A, j))) {
            weight[i]++;
            SET(GETSET(A, j), COVERED);
          }
        }
      }
    }

    /* Print out the contraints */
    if (!output_symbolic) {
      (void)fprintf(
          fp, "# Symbolic constraints for variable %d (Numeric form)\n", var);
      (void)fprintf(fp, "# unconstrained weight = %d\n", noweight);
      (void)fprintf(fp, "num_codes=%d\n", cube.part_size[var]);
      for (i = 0; i < A->count; i++) {
        if (weight[i] > 0) {
          (void)fprintf(fp, "weight=%d: ", weight[i]);
          for (j = 0; j < A->sf_size; j++) {
            if (is_in_set(GETSET(A, i), j)) {
              (void)fprintf(fp, " %d", j);
            }
          }
          (void)fprintf(fp, "\n");
        }
      }
    } else {
      (void)fprintf(
          fp, "# Symbolic constraints for variable %d (Symbolic form)\n", var);
      for (i = 0; i < A->count; i++) {
        if (weight[i] > 0) {
          (void)fprintf(fp, "#   w=%d: (", weight[i]);
          for (j = 0; j < A->sf_size; j++) {
            if (is_in_set(GETSET(A, i), j)) {
              (void)fprintf(fp, " %s", PLA->label[cube.first_part[var] + j]);
            }
          }
          (void)fprintf(fp, " )\n");
        }
      }
      FREE(weight);
    }
  }
}
