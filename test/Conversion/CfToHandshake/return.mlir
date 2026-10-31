// RUN: dynamatic-opt --func-merge-returns --lower-cf-to-handshake --remove-operation-names %s --split-input-file | FileCheck %s

// a returned argument goes straight to the end
// CHECK-LABEL: handshake.func @simpleReturn(
// CHECK-SAME:    %[[IN0:[a-zA-Z0-9_]+]]: !handshake.channel<i32>, %[[START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         end {{.*}} %[[IN0]], %[[START]] :
func.func @simpleReturn(%arg0 : i32 {handshake.arg_name = "in0"}) -> i32 {
  return %arg0 : i32
}

// -----

// each returned value is its own end input, in order
// CHECK-LABEL: handshake.func @retunMultipleValues(
// CHECK-SAME:    %[[IN0:[a-zA-Z0-9_]+]]: !handshake.channel<i32>, %[[IN1:[a-zA-Z0-9_]+]]: !handshake.channel<i1>, %[[IN2:[a-zA-Z0-9_]+]]: !handshake.channel<i32>, %[[START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         end {{.*}} %[[IN0]], %[[IN1]], %[[IN2]], %[[START]] :
func.func @retunMultipleValues(%arg0 : i32 {handshake.arg_name = "in0"}, %arg1 : i1 {handshake.arg_name = "in1"}, %arg2 : index {handshake.arg_name = "in2"}) -> (i32, i1, index) {
  return %arg0, %arg1, %arg2 : i32, i1, index
}

// -----

// the two returns become one end,
// whose value is muxed between the two paths
// CHECK-LABEL: handshake.func @multipleReturns(
// CHECK:         %[[MUX:[a-zA-Z0-9_]+]] = mux %[[INDEX:[a-zA-Z0-9_]+]] {{\[}}
// CHECK:         %{{[a-zA-Z0-9_]+}}, %[[INDEX]] = control_merge
// CHECK:         end {{.*}} %[[MUX]],
// CHECK-NOT:     end
func.func @multipleReturns(%arg0 : i1 {handshake.arg_name = "in0"}) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  return %c0 : i32
^bb2:
  return %c1 : i32
}

// -----

// memories with no accesses pass their start straight to their end
// CHECK-LABEL: handshake.func @memoryConnect(
// CHECK-SAME:    %[[MEM0_START:[a-zA-Z0-9_]+]]: !handshake.control<>, %[[MEM1_START:[a-zA-Z0-9_]+]]: !handshake.control<>, %[[START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         end {{.*}}, %[[MEM0_START]], %[[MEM1_START]], %[[START]] :
func.func @memoryConnect(%arg0: i1 {handshake.arg_name = "in0"}, %arg1: memref<4xi32> {handshake.arg_name = "mem0"}, %arg2: memref<4xi32> {handshake.arg_name = "mem1"}) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  return %c0 : i32
^bb2:
  return %c1 : i32
}
