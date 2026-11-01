// RUN: dynamatic-opt %s --func-merge-returns | FileCheck %s

// A function with one return is left as it is
// CHECK-LABEL: func.func @single_return
func.func @single_return(%arg0: i32) -> i32 {
  // CHECK-NEXT: return %arg0
  return %arg0 : i32
}

// A function with two returns
// CHECK-LABEL: func.func @two_returns
func.func @two_returns(%arg0: i1, %arg1: i32, %arg2: i32) -> (i32, i32) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // this return becomes a branch to the new exit block,
  // passing %arg1 and %arg2
  // CHECK: cf.br ^[[EXIT:.*]](%arg1, %arg2 : i32, i32)
  return %arg1, %arg2 : i32, i32
^bb2:
  // this return becomes a branch to the same exit block,
  // passing %arg2 and %arg1
  // CHECK: cf.br ^[[EXIT]](%arg2, %arg1 : i32, i32)
  return %arg2, %arg1 : i32, i32
  // the new exit block returns the values it was passed, in order
  // CHECK: ^[[EXIT]](%[[FIRST:.*]]: i32, %[[SECOND:.*]]: i32):
  // CHECK-NEXT: return %[[FIRST]], %[[SECOND]]
}

// A function with two returns and no returned values
// CHECK-LABEL: func.func @no_values
func.func @no_values(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // this return becomes a branch to the new exit block
  // CHECK: cf.br ^[[EXIT:.*]]
  return
^bb2:
  // this return becomes a branch to the same exit block
  // CHECK: cf.br ^[[EXIT]]
  return
  // the new exit block returns
  // CHECK: ^[[EXIT]]:
  // CHECK-NEXT: return
}
