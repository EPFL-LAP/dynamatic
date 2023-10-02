//===- binary_search.c - Search for integer in array  -------------*- C -*-===//
//
// Implements the binary_search kernel.
//
//===----------------------------------------------------------------------===//

#include "binary_search.h"

int binary_search(int search, int a[N]) {
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
