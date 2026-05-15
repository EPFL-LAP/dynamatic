// RUN: %source-rewriter && FileCheck %s --input-file=%t.c

// CHECK: return ((((((a) != 0) & ((b) != 0))) != 0) & ((c) != 0));
int nested_and(int a, int b, int c) { return a && b && c; }

// CHECK: return ((((((a) != 0) & ((b) != 0))) != 0) | ((c) != 0));
int mixed(int a, int b, int c) { return a && b || c; }
