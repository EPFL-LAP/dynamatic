//===- binary_search.c - Search for integer in array  -------------*- C -*-===//
//
// Implements the binary_search kernel.
//
//===----------------------------------------------------------------------===//

#include "binary_search.h"
#include "../integration_utils.h"

int binary_search(in_int_t search, in_int_t a[N]) {
  int evenIdx = -1;
  int oddIdx = -1;

  for (unsigned i = 0; i < N; i += 2) {
    if (a[i] == search) {
      evenIdx = (int)i;
      break;
    }
  }

  for (unsigned i = 1; i < N; i += 2) {
    if (a[i] == search) {
      oddIdx = (int)i;
      break;
    }
  }

  int done = -1;
  if (evenIdx != -1)
    done = evenIdx;
  else if (oddIdx != -1)
    done = oddIdx;

  return done;
}

int main(void) {
  in_int_t a[N];
  for (int i = 0; i < N; i++)
    a[i] = i;
  CALL_KERNEL(binary_search, 55, a);
  return 0;
}
