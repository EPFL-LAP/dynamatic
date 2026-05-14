// RUN: %no-short-circuit && FileCheck %s --input-file=%t.c

// CHECK: return (!!(a) & !!(b));
int land(int a, int b) { return a && b; }

// CHECK: return (!!(a) | !!(b));
int lor(int a, int b) { return a || b; }
