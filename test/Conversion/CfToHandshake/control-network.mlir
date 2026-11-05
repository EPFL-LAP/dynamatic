// RUN: dynamatic-opt %s --lower-cf-to-handshake --remove-operation-names --split-input-file | FileCheck %s

// An if-else where each branch produces the value used after it
// CHECK-LABEL: handshake.func @if_else
func.func @if_else(%arg0: i1 {handshake.arg_name = "in0"}) -> i32 {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // capture the bb1 constant, to check the order of the mux inputs
  // CHECK-DAG: %[[C1:[^ ]+]] = constant {{.*}}value = 1
  %c1 = arith.constant 1 : i32
  cf.br ^bb3(%c1 : i32)
^bb2:
  // capture the bb2 constant, to check the order of the mux inputs
  // CHECK-DAG: %[[C2:[^ ]+]] = constant {{.*}}value = 2
  %c2 = arith.constant 2 : i32
  cf.br ^bb3(%c2 : i32)
^bb3(%x: i32):
  // an init
  // CHECK-DAG: %[[INIT:[^ ]+]] = init %[[INIT_INPUT:[^ ]+]]
  // a cond_br on in0, whose data input is the output of the init
  // CHECK-DAG: %[[INIT_TO_BB1:[^ ]+]], %[[INIT_TO_BB2:[^ ]+]] = cond_br %in0, %[[INIT]]
  // a control merge of the outputs of the cond_br to bb1 and to bb2,
  // whose non-select output is the input of the init
  // CHECK-DAG: %[[INIT_INPUT]], %[[SELECT:[^ ]+]] = control_merge [%[[INIT_TO_BB1]], %[[INIT_TO_BB2]]]
  // a mux whose select input is the select output of the control merge,
  // taking the bb1 constant then the bb2 constant
  // CHECK-DAG: mux %[[SELECT]] [%[[C1]], %[[C2]]]
  return %x : i32
}

// -----

// An if-else producing a value, then an unrelated loop containing an if-else,
// then a return of the value
// CHECK-LABEL: handshake.func @if_else_then_loop
func.func @if_else_then_loop(%arg0: i1 {handshake.arg_name = "in0"},
                             %arg1: i1 {handshake.arg_name = "in1"}) -> i32 {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // capture the bb1 constant, to check the order of the mux inputs
  // CHECK-DAG: %[[C100:[^ ]+]] = constant {{.*}}value = 100
  %c100 = arith.constant 100 : i32
  cf.br ^bb3(%c100 : i32)
^bb2:
  // capture the bb2 constant, to check the order of the mux inputs
  // CHECK-DAG: %[[C200:[^ ]+]] = constant {{.*}}value = 200
  %c200 = arith.constant 200 : i32
  cf.br ^bb3(%c200 : i32)
^bb3(%x: i32):
  %c0 = arith.constant 0 : i32
  cf.br ^bb4(%c0 : i32)
^bb4(%i: i32):
  %c10 = arith.constant 10 : i32
  %cond = arith.cmpi slt, %i, %c10 : i32
  cf.cond_br %cond, ^bb5, ^bb9
^bb5:
  cf.cond_br %arg1, ^bb6, ^bb7
^bb6:
  cf.br ^bb8
^bb7:
  cf.br ^bb8
^bb8:
  %c1 = arith.constant 1 : i32
  // the loop still exists: the output of the addi in bb8
  // is an input of a mux in bb4
  // CHECK-DAG: %[[NEXT:[^ ]+]] = addi {{.*}} {handshake.bb = 8
  // CHECK-DAG: mux {{.*}}%[[NEXT]]] {handshake.bb = 4
  %next = arith.addi %i, %c1 : i32
  cf.br ^bb4(%next : i32)
^bb9:
  // the routing of %x depends only on the if-else on in0:
  // an init
  // CHECK-DAG: %[[INIT:[^ ]+]] = init %[[INIT_INPUT:[^ ]+]]
  // a cond_br on in0, whose data input is the output of the init
  // CHECK-DAG: %[[INIT_TO_BB1:[^ ]+]], %[[INIT_TO_BB2:[^ ]+]] = cond_br %in0, %[[INIT]]
  // a control merge of the outputs of the cond_br to bb1 and to bb2,
  // whose non-select output is the input of the init
  // CHECK-DAG: %[[INIT_INPUT]], %[[SELECT:[^ ]+]] = control_merge [%[[INIT_TO_BB1]], %[[INIT_TO_BB2]]]
  // a mux whose select input is the select output of the control merge,
  // taking the bb1 constant then the bb2 constant
  // CHECK-DAG: %[[MUX:[^ ]+]] = mux %[[SELECT]] [%[[C100]], %[[C200]]]
  // the output of the mux is an input of the end
  // CHECK-DAG: end {{.*}} %[[MUX]],
  return %x : i32
}
