// RUN: %source-rewriter && FileCheck %s --input-file=%t.c

// CHECK: return (!!((!!(a) & !!(b))) & !!(c));
int nested_and(int a, int b, int c) { return a && b && c; }

// CHECK: return (!!((!!(a) & !!(b))) | !!(c));
int mixed(int a, int b, int c) { return a && b || c; }
