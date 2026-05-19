// RUN: not %dyn-clang-pragmas %s | FileCheck %s

// CHECK: #pragma DYN speculate requires variable=, max_predictions=, style=
int main(void) {
  int x = 0;
  #pragma DYN speculate variable=x
  return x;
}
