#include "matching.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

int matching(in_int_t edges[1000], inout_float_t vertices[1000],
               in_int_t num_edges) {

  int i = 0;
  int out = 0;

  while (i < num_edges) {

    int j = i * 2;

    int u = edges[j];
    int v = edges[j + 1];

    float t1 = vertices[u];
    float t2 = vertices[v];

    if (t1 * t2 == 1) {
    // if ((t1 < 0) && (t2 < 0)) {
      vertices[u] = v;
      vertices[v] = u;

      out = out + 1;
    }

    i = i + 1;
  }

  return out;
}

int main(void) {
  in_int_t edges[1000];
  inout_float_t vertices[1000];
  in_int_t num_edges;

  for (int i = 0; i < 1000; ++i) {
    edges[i] = (i + 10) % 1000;
    vertices[i] = (rand() % 2) * 2 - 1;
  }
  num_edges = 400;

  CALL_KERNEL(matching, edges, vertices, num_edges);
}
