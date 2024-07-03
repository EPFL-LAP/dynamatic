/*
    module: cvrin.c
    purpose: cube and cover input routines
*/

#include "dynamatic/Support/Espresso/espresso.h"
#include "dynamatic/Support/Espresso/strdup.h"
#include <ctype.h>

static Bool line_length_error;
static int lineno;

void skip_line(FILE *fpin, FILE *fpout, int echo) {
  int ch;
  while ((ch = getc(fpin)) != EOF && ch != '\n')
    if (echo)
      putc(ch, fpout);
  if (echo)
    putc('\n', fpout);
  lineno++;
}

char *get_word(FILE *fp, char *word) {
  int ch, i = 0;
  while ((ch = getc(fp)) != EOF && isspace(ch))
    ;
  word[i++] = ch;
  while ((ch = getc(fp)) != EOF && !isspace(ch))
    word[i++] = ch;
  word[i++] = '\0';
  return word;
}

//------Defining the StringStream------

// Initialize the StringStream
void initStringStream(StringStream *ss, const char *str) {
  ss->s = strdup(str); // Copy the input string
  ss->i = 0;           // Start position at the beginning of the string
}

// Free the allocated memory for the string
void freeStringStream(StringStream *ss) { free(ss->s); }

// Function to get a character from the StringStream
int stringStreamGetc(StringStream *ss) {
  if (ss->s[ss->i] == '\0') {
    return EOF; // End of string
  }
  return ss->s[ss->i++];
}

// Function to unget a character (move the position back)
int stringStreamUngetc(StringStream *ss) {
  if (ss->i > 0) {
    ss->i--;
    return ss->s[ss->i];
  }
  return EOF; // Beginning of string
}

void stringStreamSkipLine(StringStream *ss) {
  int ch;
  while ((ch = stringStreamGetc(ss)) != EOF && ch != '\n')
    lineno++;
}

char *stringStreamGetWord(StringStream *ss, char *word) {
  int ch, i = 0;
  while ((ch = stringStreamGetc(ss)) != EOF && isspace(ch))
    ;
  word[i++] = ch;
  while ((ch = stringStreamGetc(ss)) != EOF && !isspace(ch))
    word[i++] = ch;
  word[i++] = '\0';
  return word;
}

// Custom function to read an integer from the StringStream
int stringStreamScanf(StringStream *ss, int *value) {
  int ch;
  int result = 0;
  int sign = 1;

  // Skip any whitespace
  while ((ch = stringStreamGetc(ss)) != EOF && isspace(ch))
    ;

  // Check for sign
  if (ch == '-') {
    sign = -1;
    ch = stringStreamGetc(ss);
  } else if (ch == '+') {
    ch = stringStreamGetc(ss);
  }

  // Read the integer part
  if (!isdigit(ch)) {
    return 0; // Not a valid integer
  }

  do {
    result = result * 10 + (ch - '0');
    ch = stringStreamGetc(ss);
  } while (ch != EOF && isdigit(ch));

  // Move back one character if we have read one past the number
  if (ch != EOF) {
    stringStreamUngetc(ss);
  }

  *value = result * sign;
  return 1; // Successfully read one integer
}

/*
 *  Yes, I know this routine is a mess
 */
void read_cube(StringStream *ss, pPLA PLA) {
  int var, i;
  pcube cf = cube.temp[0], cr = cube.temp[1], cd = cube.temp[2];
  Bool savef = FALSE, saved = FALSE, saver = FALSE;
  char token[256];               /* for kiss read hack */
  int varx, first, last, offset; /* for kiss read hack */

  set_clear(cf, cube.size);

  /* Loop and read binary variables */
  for (var = 0; var < cube.num_binary_vars; var++) {
    switch (stringStreamGetc(ss)) {
    case EOF:
      goto bad_char;
    case '\n':
      if (!line_length_error)
        fprintf(stderr,
                "product term(s) span more than one line (warning only)\n");
      line_length_error = TRUE;
      lineno++;
      var--;
      break;
    case ' ':
    case '|':
    case '\t':
      var--;
      break;
    case '2':
    case '-':
      set_insert(cf, var * 2 + 1);
      [[fallthrough]];
    case '0':
      set_insert(cf, var * 2);
      break;
    case '1':
      set_insert(cf, var * 2 + 1);
      break;
    case '?':
      break;
    default:
      goto bad_char;
    }
  }

  /* Loop for the all but one of the multiple-valued variables */
  for (var = cube.num_binary_vars; var < cube.num_vars - 1; var++) {
    /* Read a symbolic multiple-valued variable */
    if (cube.part_size[var] < 0) {

      if (equal(token, "-") || equal(token, "ANY")) {
        if (kiss && var == cube.num_vars - 2) {
          /* leave it empty */
        } else {
          /* make it full */
          set_or(cf, cf, cube.var_mask[var]);
        }
      } else if (equal(token, "~")) {
        ;
        /* leave it empty ... (?) */
      } else {
        if (kiss && var == cube.num_vars - 2)
          varx = var - 1, offset = ABS(cube.part_size[var - 1]);
        else
          varx = var, offset = 0;

        /* Find the symbolic label in the label table */
        first = cube.first_part[varx];
        last = cube.last_part[varx];

        for (i = first; i <= last; i++) {
          if (PLA->label[i] == (char *)NULL) {
            PLA->label[i] = strdup(token); /* add new label */
            set_insert(cf, i + offset);
            break;
          } else if (equal(PLA->label[i], token)) {
            set_insert(cf, i + offset); /* use column i */
            break;
          }
        }

        if (i > last) {
          fprintf(stderr,
                  "declared size of variable %d (counting from variable 0) is "
                  "too small\n",
                  var);
          exit(-1);
        }
      }
    } else {
      for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
        switch (stringStreamGetc(ss)) {
        case EOF:
          goto bad_char;
        case '\n':
          if (!line_length_error)
            fprintf(stderr,
                    "product term(s) span more than one line (warning only)\n");

          line_length_error = TRUE;
          lineno++;
          i--;
          break;
        case ' ':
        case '|':
        case '\t':
          i--;
          break;
        case '1':
          set_insert(cf, i);
          break;
        case '0':
          break;
        default:
          goto bad_char;
        }
      }
    }
  }

  /* Loop for last multiple-valued variable */
  if (kiss) {
    saver = savef = TRUE;
    (void)set_xor(cr, cf, cube.var_mask[cube.num_vars - 2]);
  } else {
    set_copy(cr, cf);
  }

  set_copy(cd, cf);

  for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
    switch (stringStreamGetc(ss)) {
    case EOF:
      goto bad_char;
    case '\n':
      if (!line_length_error)
        fprintf(stderr,
                "product term(s) span more than one line (warning only)\n");
      line_length_error = TRUE;
      lineno++;
      i--;
      break;
    case ' ':
    case '|':
    case '\t':
      i--;
      break;
    case '4':
    case '1':
      if (PLA->pla_type & F_type)
        set_insert(cf, i), savef = TRUE;
      break;
    case '3':
    case '0':
      if (PLA->pla_type & R_type)
        set_insert(cr, i), saver = TRUE;
      break;
    case '2':
    case '-':
      if (PLA->pla_type & D_type)
        set_insert(cd, i), saved = TRUE;
      break;
    case '~':
      break;
    default:
      goto bad_char;
    }
  }

  if (savef)
    PLA->F = sf_addset(PLA->F, cf);
  if (saved)
    PLA->D = sf_addset(PLA->D, cd);
  if (saver)
    PLA->R = sf_addset(PLA->R, cr);
  return;

bad_char:
  fprintf(stderr, "(warning): input line #%d ignored\n", lineno);
  stringStreamSkipLine(ss);
  return;
}

void parse_pla(StringStream *ss, pPLA PLA) {
  int i, var, ch, last;
  char word[256];

  lineno = 1;
  line_length_error = FALSE;

  while ((ch = stringStreamGetc(ss)) != EOF) {
    switch (ch) {
    case EOF:
      return;
    case '\n':
      lineno++;
      break;
    case ' ':
    case '\t':
    case '\f':
    case '\r':
      break;
    case '#':
      (void)stringStreamGetc(ss);
      stringStreamSkipLine(ss);
      break;
    case '.':
      /* .i gives the cube input size (binary-functions only) */
      if (equal(stringStreamGetWord(ss, word), "i")) {
        if (cube.fullset != NULL) {
          fprintf(stderr, "extra .i ignored\n");
          stringStreamSkipLine(ss);
        } else {
          if (stringStreamScanf(ss, &cube.num_binary_vars) != 1)
            fatal("error reading .i");

          cube.num_vars = cube.num_binary_vars + 1;
          cube.part_size = ALLOC(int, cube.num_vars);
        }
        /* .o gives the cube output size (binary-functions only) */
      } else if (equal(word, "o")) {
        if (cube.fullset != NULL) {
          fprintf(stderr, "extra .o ignored\n");
          stringStreamSkipLine(ss);
        } else {
          if (cube.part_size == NULL)
            fatal(".o cannot appear before .i");

          if (stringStreamScanf(ss, &(cube.part_size[cube.num_vars - 1])) != 1)
            fatal("error reading .o");

          cube_setup();
          PLA_labels(PLA);
        }
        /* .mv gives the cube size for a multiple-valued function */
      } else if (equal(word, "mv")) {
        if (cube.fullset != NULL) {
          fprintf(stderr, "extra .mv ignored\n");
          stringStreamSkipLine(ss);
        } else {
          if (cube.part_size != NULL)
            fatal("cannot mix .i and .mv");

          if (cube.num_binary_vars < 0)
            fatal("num_binary_vars (second field of .mv) cannot be negative");

          if (cube.num_vars < cube.num_binary_vars)
            fatal("num_vars (1st field of .mv) must exceed num_binary_vars "
                  "(2nd field of .mv)");

          cube.part_size = ALLOC(int, cube.num_vars);

          for (var = cube.num_binary_vars; var < cube.num_vars; var++)
            cube_setup();
          PLA_labels(PLA);
        }
        /* .p gives the number of product terms -- we ignore it */
      } else if (equal(word, "p")) {

      }
      //(void) fscanf(fp, "%d", &np);
      /* .e and .end specify the end of the file */
      else if (equal(word, "e") || equal(word, "end")) {
        return;
      }
      /* .kiss turns on the kiss-hack option */
      else if (equal(word, "kiss")) {
        kiss = TRUE;
      }
      /* .type specifies a logical type for the PLA */
      else if (equal(word, "type")) {
        (void)stringStreamGetWord(ss, word);

        for (i = 0; pla_types[i].key != 0; i++)
          if (equal(pla_types[i].key + 1, word)) {
            PLA->pla_type = pla_types[i].value;
            break;
          }

        if (pla_types[i].key == 0)
          fatal("unknown type in .type command");
        /* parse the labels */
      } else if (equal(word, "ilb")) {
        if (cube.fullset == NULL)
          fatal("PLA size must be declared before .ilb or .ob");

        if (PLA->label == NULL)
          PLA_labels(PLA);

        for (var = 0; var < cube.num_binary_vars; var++) {
          (void)stringStreamGetWord(ss, word);
          i = cube.first_part[var];
          PLA->label[i + 1] = strdup(word);
          PLA->label[i] = ALLOC(char, strlen(word) + 6);
          (void)sprintf(PLA->label[i], "%s.bar", word);
        }
      } else if (equal(word, "ob")) {
        if (cube.fullset == NULL)
          fatal("PLA size must be declared before .ilb or .ob");

        if (PLA->label == NULL)
          PLA_labels(PLA);

        var = cube.num_vars - 1;
        for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
          (void)stringStreamGetWord(ss, word);
          PLA->label[i] = strdup(word);
        }
        /* .label assigns labels to multiple-valued variables */
      } else if (equal(word, "label")) {
        if (cube.fullset == NULL)
          fatal("PLA size must be declared before .label");

        if (PLA->label == NULL)
          PLA_labels(PLA);

        for (i = cube.first_part[var]; i <= cube.last_part[var]; i++) {
          (void)stringStreamGetWord(ss, word);
          PLA->label[i] = strdup(word);
        }
      } else if (equal(word, "symbolic")) {
        symbolic_t *newlist, *p1;

        if (read_symbolic(ss, PLA, word, &newlist)) {
          if (PLA->symbolic == NIL(symbolic_t)) {
            PLA->symbolic = newlist;
          } else {
            for (p1 = PLA->symbolic; p1->next != NIL(symbolic_t); p1 = p1->next)
              ;
            p1->next = newlist;
          }
        } else {
          fatal("error reading .symbolic");
        }
      } else if (equal(word, "symbolic-output")) {
        symbolic_t *newlist, *p1;

        if (read_symbolic(ss, PLA, word, &newlist)) {
          if (PLA->symbolic_output == NIL(symbolic_t)) {
            PLA->symbolic_output = newlist;
          } else {
            for (p1 = PLA->symbolic_output; p1->next != NIL(symbolic_t);
                 p1 = p1->next)
              ;
            p1->next = newlist;
          }
        } else {
          fatal("error reading .symbolic-output");
        }
        /* .phase allows a choice of output phases */
      } else if (equal(word, "phase")) {
        if (cube.fullset == NULL)
          fatal("PLA size must be declared before .phase");

        if (PLA->phase != NULL) {
          fprintf(stderr, "extra .phase ignored\n");
          stringStreamSkipLine(ss);
        } else {
          do
            ch = stringStreamGetc(ss);
          while (ch == ' ' || ch == '\t');
          (void)stringStreamUngetc(ss);
          PLA->phase = set_save(cube.fullset);
          last = cube.last_part[cube.num_vars - 1];

          for (i = cube.first_part[cube.num_vars - 1]; i <= last; i++)
            if ((ch = stringStreamGetc(ss)) == '0')
              set_remove(PLA->phase, i);
            else if (ch != '1')
              fatal("only 0 or 1 allowed in phase description");
        }
        /* .pair allows for bit-pairing input variables */
      } else if (equal(word, "pair")) {
        int j;

        if (PLA->pair != NULL) {
          fprintf(stderr, "extra .pair ignored\n");
        } else {
          ppair pair;
          PLA->pair = pair = ALLOC(pair_t, 1);

          pair->var1 = ALLOC(int, pair->cnt);
          pair->var2 = ALLOC(int, pair->cnt);

          for (i = 0; i < pair->cnt; i++) {
            (void)stringStreamGetWord(ss, word);

            if (word[0] == '(')
              (void)strcpy(word, word + 1);

            if (label_index(PLA, word, &var, &j)) {
              pair->var1[i] = var + 1;
            } else {
              fatal("syntax error in .pair");
            }

            (void)stringStreamGetWord(ss, word);

            if (word[strlen(word) - 1] == ')') {
              word[strlen(word) - 1] = '\0';
            }

            if (label_index(PLA, word, &var, &j)) {
              pair->var2[i] = var + 1;
            } else {
              fatal("syntax error in .pair");
            }
          }
        }
      } else {
        if (echo_unknown_commands)
          printf("%c%s ", ch, word);

        stringStreamSkipLine(ss);
      }
      break;
    default:

      (void)stringStreamUngetc(ss);

      if (cube.fullset == NULL) {
        /*		fatal("unknown PLA size, need .i/.o or .mv");*/
        if (echo_comments)
          putchar('#');
        stringStreamSkipLine(ss);
        break;
      }

      if (PLA->F == NULL) {
        PLA->F = new_cover(10);
        PLA->D = new_cover(10);
        PLA->R = new_cover(10);
      }

      read_cube(ss, PLA);
    }
  }
}

/*
    read_pla -- read a PLA from a file

    Input stops when ".e" is encountered in the input file, or upon reaching
    end of file.

    Returns the PLA in the variable PLA after massaging the "symbolic"
    representation into a positional cube notation of the ON-set, OFF-set,
    and the DC-set.

    needs_dcset and needs_offset control the computation of the OFF-set
    and DC-set (i.e., if either needs to be computed, then it will be
    computed via complement only if the corresponding option is TRUE.)
    pla_type specifies the interpretation to be used when reading the
    PLA.

    The phase of the output functions is adjusted according to the
    global option "pos" or according to an imbedded .phase option in
    the input file.  Note that either phase option implies that the
    OFF-set be computed regardless of whether the caller needs it
    explicitly or not.

    Bit pairing of the binary variables is performed according to an
    imbedded .pair option in the input file.

    The global cube structure also reflects the sizes of the PLA which
    was just read.  If these fields have already been set, then any
    subsequent PLA must conform to these sizes.

    The global flags trace and summary control the output produced
    during the read.

    Returns a status code as a result:
  EOF (-1) : End of file reached before any data was read
  > 0	 : Operation successful
*/

int read_pla(char *fp, int needs_dcset, int needs_offset, int pla_type,
             pPLA *PLA_return) {
  pPLA PLA;
  int i;
  long time;
  cost_t cost;

  /* Allocate and initialize the PLA structure */
  PLA = *PLA_return = new_PLA();
  PLA->pla_type = pla_type;

  /* Read the pla */
  time = ptime();
  StringStream ss;
  initStringStream(&ss, fp);
  parse_pla(&ss, PLA);

  /* Check for nothing on the file -- implies reached EOF */
  if (PLA->F == NULL) {
    return EOF;
  }

  /* This hack merges the next-state field with the outputs */
  for (i = 0; i < cube.num_vars; i++) {
    cube.part_size[i] = ABS(cube.part_size[i]);
  }

  /* Decide how to break PLA into ON-set, OFF-set and DC-set */
  time = ptime();
  if (pos || PLA->phase != NULL || PLA->symbolic_output != NIL(symbolic_t)) {
    needs_offset = TRUE;
  }

  if (needs_offset && (PLA->pla_type == F_type || PLA->pla_type == FD_type)) {
    free_cover(PLA->R);
    PLA->R = complement(cube2list(PLA->F, PLA->D));
  } else if (needs_dcset && PLA->pla_type == FR_type) {
    pcover X;
    free_cover(PLA->D);

    /* hack, why not? */
    X = d1merge(sf_join(PLA->F, PLA->R), cube.num_vars - 1);
    PLA->D = complement(cube1list(X));
    free_cover(X);
  } else if (PLA->pla_type == R_type || PLA->pla_type == DR_type) {
    free_cover(PLA->F);
    PLA->F = complement(cube2list(PLA->D, PLA->R));
  }

  if (trace) {
    totals(time, COMPL_TIME, PLA->R, &cost);
  }

  /* Check for phase rearrangement of the functions */
  if (pos) {
    pcover onset = PLA->F;
    PLA->F = PLA->R;
    PLA->R = onset;
    PLA->phase = new_cube();
    set_diff(PLA->phase, cube.fullset, cube.var_mask[cube.num_vars - 1]);
  } else if (PLA->phase != NULL) {
    (void)set_phase(PLA);
  }

  /* Setup minimization for two-bit decoders */
  if (PLA->pair != (ppair)NULL) {
    set_pair(PLA);
  }

  if (PLA->symbolic != NIL(symbolic_t)) {
    EXEC(map_symbolic(PLA), "MAP-INPUT  ", PLA->F);
  }

  if (PLA->symbolic_output != NIL(symbolic_t)) {
    EXEC(map_output_symbolic(PLA), "MAP-OUTPUT ", PLA->F);

    if (needs_offset) {
      free_cover(PLA->R);
      EXECUTE(PLA->R = complement(cube2list(PLA->F, PLA->D)), COMPL_TIME,
              PLA->R, cost);
    }
  }

  return 1;
}

void PLA_summary(pPLA PLA) {
  int var, i;
  symbolic_list_t *p2;
  symbolic_t *p1;

  printf("# PLA is %s", PLA->filename);
  if (cube.num_binary_vars == cube.num_vars - 1)
    printf(" with %d inputs and %d outputs\n", cube.num_binary_vars,
           cube.part_size[cube.num_vars - 1]);
  else {
    printf(" with %d variables (%d binary, mv sizes", cube.num_vars,
           cube.num_binary_vars);
    for (var = cube.num_binary_vars; var < cube.num_vars; var++)
      printf(" %d", cube.part_size[var]);
    printf(")\n");
  }
  printf("# ON-set cost is  %s\n", print_cost(PLA->F));
  printf("# OFF-set cost is %s\n", print_cost(PLA->R));
  printf("# DC-set cost is  %s\n", print_cost(PLA->D));
  if (PLA->phase != NULL)
    printf("# phase is %s\n", pc1(PLA->phase));
  if (PLA->pair != NULL) {
    printf("# two-bit decoders:");
    for (i = 0; i < PLA->pair->cnt; i++)
      printf(" (%d %d)", PLA->pair->var1[i], PLA->pair->var2[i]);
    printf("\n");
  }
  if (PLA->symbolic != NIL(symbolic_t)) {
    for (p1 = PLA->symbolic; p1 != NIL(symbolic_t); p1 = p1->next) {
      printf("# symbolic: ");
      for (p2 = p1->symbolic_list; p2 != NIL(symbolic_list_t); p2 = p2->next) {
        printf(" %d", p2->variable);
      }
      printf("\n");
    }
  }
  if (PLA->symbolic_output != NIL(symbolic_t)) {
    for (p1 = PLA->symbolic_output; p1 != NIL(symbolic_t); p1 = p1->next) {
      printf("# output symbolic: ");
      for (p2 = p1->symbolic_list; p2 != NIL(symbolic_list_t); p2 = p2->next) {
        printf(" %d", p2->pos);
      }
      printf("\n");
    }
  }
  (void)fflush(stdout);
}

pPLA new_PLA(void) {
  pPLA PLA;

  PLA = ALLOC(PLA_t, 1);
  PLA->F = PLA->D = PLA->R = (pcover)NULL;
  PLA->phase = (pcube)NULL;
  PLA->pair = (ppair)NULL;
  PLA->label = (char **)NULL;
  PLA->filename = (char *)NULL;
  PLA->pla_type = 0;
  PLA->symbolic = NIL(symbolic_t);
  PLA->symbolic_output = NIL(symbolic_t);
  return PLA;
}

void PLA_labels(pPLA PLA) {
  int i;

  PLA->label = ALLOC(char *, cube.size);
  for (i = 0; i < cube.size; i++)
    PLA->label[i] = (char *)NULL;
}

void free_PLA(pPLA PLA) {
  symbolic_list_t *p2, *p2next;
  symbolic_t *p1, *p1next;
  int i;

  if (PLA->F != (pcover)NULL)
    free_cover(PLA->F);

  if (PLA->R != (pcover)NULL)
    free_cover(PLA->R);

  if (PLA->D != (pcover)NULL)
    free_cover(PLA->D);

  if (PLA->phase != (pcube)NULL)
    free_cube(PLA->phase);

  if (PLA->pair != (ppair)NULL) {
    FREE(PLA->pair->var1);
    FREE(PLA->pair->var2);
    FREE(PLA->pair);
  }

  if (PLA->label != NULL) {
    for (i = 0; i < cube.size; i++)
      if (PLA->label[i] != NULL)
        FREE(PLA->label[i]);

    FREE(PLA->label);
  }

  if (PLA->filename != NULL) {
    FREE(PLA->filename);
  }

  for (p1 = PLA->symbolic; p1 != NIL(symbolic_t); p1 = p1next) {
    for (p2 = p1->symbolic_list; p2 != NIL(symbolic_list_t); p2 = p2next) {
      p2next = p2->next;
      FREE(p2);
    }

    p1next = p1->next;
    FREE(p1);
  }

  PLA->symbolic = NIL(symbolic_t);

  for (p1 = PLA->symbolic_output; p1 != NIL(symbolic_t); p1 = p1next) {
    for (p2 = p1->symbolic_list; p2 != NIL(symbolic_list_t); p2 = p2next) {
      p2next = p2->next;
      FREE(p2);
    }

    p1next = p1->next;
    FREE(p1);
  }

  PLA->symbolic_output = NIL(symbolic_t);
  FREE(PLA);
}

int read_symbolic(StringStream *ss, pPLA PLA, char *word, symbolic_t **retval)

/* scratch string for words */

{
  symbolic_list_t *listp, *prev_listp;
  symbolic_label_t *labelp, *prev_labelp;
  symbolic_t *newlist;
  int i, var;

  newlist = ALLOC(symbolic_t, 1);
  newlist->next = NIL(symbolic_t);
  newlist->symbolic_list = NIL(symbolic_list_t);
  newlist->symbolic_list_length = 0;
  newlist->symbolic_label = NIL(symbolic_label_t);
  newlist->symbolic_label_length = 0;
  prev_listp = NIL(symbolic_list_t);
  prev_labelp = NIL(symbolic_label_t);

  for (;;) {
    (void)stringStreamGetWord(ss, word);
    if (equal(word, ";"))
      break;
    if (label_index(PLA, word, &var, &i)) {
      listp = ALLOC(symbolic_list_t, 1);
      listp->variable = var;
      listp->pos = i;
      listp->next = NIL(symbolic_list_t);
      if (prev_listp == NIL(symbolic_list_t)) {
        newlist->symbolic_list = listp;
      } else {
        prev_listp->next = listp;
      }
      prev_listp = listp;
      newlist->symbolic_list_length++;
    } else {
      return FALSE;
    }
  }

  for (;;) {
    (void)stringStreamGetWord(ss, word);
    if (equal(word, ";"))
      break;
    labelp = ALLOC(symbolic_label_t, 1);
    labelp->label = strdup(word);
    labelp->next = NIL(symbolic_label_t);
    if (prev_labelp == NIL(symbolic_label_t)) {
      newlist->symbolic_label = labelp;
    } else {
      prev_labelp->next = labelp;
    }
    prev_labelp = labelp;
    newlist->symbolic_label_length++;
  }

  *retval = newlist;
  return TRUE;
}

int label_index(pPLA PLA, char *word, int *varp, int *ip) {
  int var, i;

  if (PLA->label == NIL(char *) || PLA->label[0] == NIL(char)) {
    if (sscanf(word, "%d", varp) == 1) {
      *ip = *varp;
      return TRUE;
    }
  } else {
    for (var = 0; var < cube.num_vars; var++) {
      for (i = 0; i < cube.part_size[var]; i++) {
        if (equal(PLA->label[cube.first_part[var] + i], word)) {
          *varp = var;
          *ip = i;
          return TRUE;
        }
      }
    }
  }
  return FALSE;
}
