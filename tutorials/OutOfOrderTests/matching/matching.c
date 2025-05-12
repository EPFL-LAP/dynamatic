#include "matching.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

int matching(in_int_t num_edges) {

  int i = 0;
  int out = 0;

  while (i < num_edges) {

    out = out + 1;

    i = i + 1;
  }

  return out;
}

int main(void) {
  in_int_t edges[1000];
  inout_int_t vertices[1000];
  in_int_t num_edges;

  for (int i = 0; i < 1000; ++i) {
    edges[i] = (i + 10) % 1000;
    vertices[i] = 700 - i % 1000;
  }
  num_edges = 400;

  CALL_KERNEL(matching, num_edges);
}