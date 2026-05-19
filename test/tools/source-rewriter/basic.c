// RUN: %source-rewriter && FileCheck %s --input-file=%t.c

// CHECK: return (((a) != 0) & ((b) != 0));
int land(int a, int b) { return a && b; }

// CHECK: return (((a) != 0) | ((b) != 0));
int lor(int a, int b) { return a || b; }
