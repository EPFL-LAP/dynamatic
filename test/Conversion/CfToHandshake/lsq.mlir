// RUN: dynamatic-opt %s --lower-cf-to-handshake --remove-operation-names | FileCheck %s

// An if-else with one store in the true branch
// and two stores in the false branch
// CHECK-LABEL: handshake.func @if_else_stores
func.func @if_else_stores(%arg0: i1 {handshake.arg_name = "in0"},
                          %arg1: memref<4xi32> {handshake.arg_name = "mem0"}) {
  // the start of mem0 (%arg2) produces the allocation token,
  // which goes through a branch on %arg0 to bb1 or bb2
  // CHECK-DAG: %[[MEM0_START_TO_BB1:[^ ]+]], %[[MEM0_START_TO_BB2:[^ ]+]] = cond_br %arg0, %arg2
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // in bb1, the allocation token goes into a lazy fork
  // CHECK-DAG: %[[BB1_LAZY_FORK:[0-9]+]]:2 = lazy_fork [2] %[[MEM0_START_TO_BB1]]
  %i0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32
  memref.store %c42, %arg1[%i0] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<4xi32>
  cf.br ^bb3
^bb2:
  // in bb2, the allocation token goes into a lazy fork
  // CHECK-DAG: %[[BB2_LAZY_FORK:[0-9]+]]:2 = lazy_fork [2] %[[MEM0_START_TO_BB2]]
  %i1 = arith.constant 1 : index
  %i3 = arith.constant 3 : index
  %c7 = arith.constant 7 : i32
  %c9 = arith.constant 9 : i32
  memref.store %c7, %arg1[%i1] {handshake.name = "store1", handshake.mem_interface = #handshake.mem_interface<LSQ: 1>} : memref<4xi32>
  memref.store %c9, %arg1[%i3] {handshake.name = "store2", handshake.mem_interface = #handshake.mem_interface<LSQ: 1>} : memref<4xi32>
  cf.br ^bb3
^bb3:
  // the second outputs of the two lazy forks merge in bb3
  // CHECK-DAG: %[[END_INPUT:[^ ]+]] = merge %[[BB1_LAZY_FORK]]#1, %[[BB2_LAZY_FORK]]#1
  // the LSQ takes the start of mem0,
  // the first output of the bb1 lazy fork to allocate the bb1 group,
  // the first output of the bb2 lazy fork to allocate the bb2 group,
  // and the output of the merge as its end input
  // CHECK-DAG: lsq[%arg1 : memref<4xi32>] start(%arg2) groups(%[[BB1_LAZY_FORK]]#0: {{.*}}, %[[BB2_LAZY_FORK]]#0: {{.*}}) end(%[[END_INPUT]])
  return
}
