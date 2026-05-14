// RUN: %source-rewriter && FileCheck %s --input-file=%t.c

// CHECK: return a + b;
int no_logical(int a, int b) { return a + b; }
