// RUN: dynamatic-opt --flatten-memref-row-major %s --remove-operation-names | FileCheck %s

// CHECK-LABEL: func.func @condBranch(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: memref<6xi32>, %[[B:.*]]: memref<6xi32>, %[[I:.*]]: index, %[[J:.*]]: index)
func.func @condBranch(%cond: i1, %a: memref<2x3xi32>, %b: memref<2x3xi32>, %i: index, %j: index) {
  // CHECK: cf.cond_br %[[COND]], ^bb1(%[[A]], %[[I]] : memref<6xi32>, index), ^bb2(%[[B]], %[[J]] : memref<6xi32>, index)
  cf.cond_br %cond, ^bb1(%a, %i : memref<2x3xi32>, index), ^bb2(%b, %j : memref<2x3xi32>, index)
// CHECK: ^bb1(%{{.*}}: memref<6xi32>, %{{.*}}: index):
^bb1(%x: memref<2x3xi32>, %k: index):
  return
// CHECK: ^bb2(%{{.*}}: memref<6xi32>, %{{.*}}: index):
^bb2(%y: memref<2x3xi32>, %l: index):
  return
}
