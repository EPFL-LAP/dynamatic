//===- binary_search.h - Search for integer in array --------------*- C -*-===//
//
// Declares the binary_search kernel which searches for a specific value in an
// integer array.
//
//===----------------------------------------------------------------------===//

#ifndef BINARY_SEARCH_BINARY_SEARCH_H
#define BINARY_SEARCH_BINARY_SEARCH_H

#define N 101

typedef int in_int_t;

/// Searches for a specific value inside an array and returns its index if it is
/// found; otherwise returns -1.
int binary_search(in_int_t search, in_int_t a[N]);

#endif // BINARY_SEARCH_BINARY_SEARCH_H
