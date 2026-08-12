// RUN: not %dyn-clang-pragmas %s | FileCheck %s

// CHECK: expected integer literal after max_predictions=
int main(void) {
  int x = 0;
  #pragma DYN speculate variable=x max_predictions="ten" style="default"
  return x;
}
