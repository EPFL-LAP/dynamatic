// RUN: not %dyn-clang-pragmas %s | FileCheck %s

// CHECK: style must be "default"
int main(void) {
  int x = 0;
  #pragma DYN speculate variable=x max_predictions=8 style="other"
  return x;
}
