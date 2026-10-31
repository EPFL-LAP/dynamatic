// RUN: dynamatic-opt --func-merge-returns --lower-cf-to-handshake --remove-operation-names %s --split-input-file | FileCheck %s

// everything in a single block is in bb 0
// CHECK-LABEL: handshake.func @simpleLoad(
// CHECK:         load{{.*}} {handshake.bb = 0 : ui32
// CHECK:         end {handshake.bb = 0 : ui32}
func.func @simpleLoad(%arg0: memref<4xi32> {handshake.arg_name = "mem0"}, %arg1: index {handshake.arg_name = "in0"}) -> i32 {
  %0 = memref.load %arg0[%arg1] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  return %0 : i32
}


// -----

// each op is in the block its cf op was in,
// and the routing into a block is in that block
// CHECK-LABEL: handshake.func @ifThenElse(
// CHECK:         cond_br {{.*}} {handshake.bb = 0 : ui32}
// CHECK:         addi {{.*}} {handshake.bb = 1 : ui32
// CHECK:         addi {{.*}} {handshake.bb = 2 : ui32
// CHECK:         mux {{.*}} {handshake.bb = 3 : ui32}
// CHECK:         control_merge {{.*}} {handshake.bb = 3 : ui32}
// CHECK:         end {handshake.bb = 3 : ui32}
func.func @ifThenElse(%arg0: i32 {handshake.arg_name = "in0"}, %arg1: i1 {handshake.arg_name = "in1"}) -> i32 {
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  %res1 = arith.addi %arg0, %c1 : i32
  cf.br ^bb3(%res1 : i32)
^bb2:
  %c2 = arith.constant 2 : i32
  %res2 = arith.addi %arg0, %c2 : i32
  cf.br ^bb3(%res2 : i32)
^bb3(%res : i32):
  return %res : i32
}

// -----

// the merged return block is numbered after the original blocks
// CHECK-LABEL: handshake.func @multipleReturns(
// CHECK:         addi {{.*}} {handshake.bb = 1 : ui32
// CHECK:         addi {{.*}} {handshake.bb = 2 : ui32
// CHECK:         mux {{.*}} {handshake.bb = 3 : ui32}
// CHECK:         end {handshake.bb = 3 : ui32}
func.func @multipleReturns(%arg0: i32 {handshake.arg_name = "in0"}, %arg1: i1 {handshake.arg_name = "in1"}) -> i32 {
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  %res1 = arith.addi %arg0, %c1 : i32
  return %res1 : i32
^bb2:
  %c2 = arith.constant 2 : i32
  %res2 = arith.addi %arg0, %c2 : i32
  return %res2 : i32
}


// -----

// the loop header's muxes and branches are in bb 1,
// the loop body is in bb 2, and the exit is in bb 3
// CHECK-LABEL: handshake.func @simpleLoop(
// CHECK:         mux {{.*}} {handshake.bb = 1 : ui32}
// CHECK:         cmpi slt, {{.*}} {handshake.bb = 1 : ui32}
// CHECK:         cond_br {{.*}} {handshake.bb = 1 : ui32}
// CHECK:         addi {{.*}} {handshake.bb = 2 : ui32
// CHECK:         end {handshake.bb = 3 : ui32}
func.func @simpleLoop(%arg0: index {handshake.arg_name = "in0"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.br ^bb1(%c0 : index)
^bb1(%1: index):
  %cmp = arith.cmpi slt, %1, %arg0 : index
  cf.cond_br %cmp, ^bb2, ^bb3
^bb2:
  %2 = arith.addi %1, %c1 : index
  cf.br ^bb1(%2 : index)
^bb3:
  return
}
