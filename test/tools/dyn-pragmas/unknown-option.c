// RUN: not %dyn-clang-pragmas %s | FileCheck %s

// CHECK: unknown option in #pragma DYN speculate
int main(void) {
  int x = 0;
  #pragma DYN speculate foo=1
  return x;
}
