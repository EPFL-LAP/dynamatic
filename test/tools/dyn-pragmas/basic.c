// RUN: %dyn-clang-pragmas %s | FileCheck %s

// CHECK: c"standard\00"
// CHECK: call i32 @__dyn_speculate(double {{.+}}, i32 noundef 8, ptr noundef {{.+}})
// CHECK: declare i32 @__dyn_speculate(double noundef, i32 noundef, ptr noundef)

int compute(void);
int use(int);

int main(void) {
  int x = compute();
  #pragma DYN speculate variable=x max_predictions=8 style=standard
  return use(x);
}
