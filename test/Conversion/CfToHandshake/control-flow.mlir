// RUN: dynamatic-opt --func-merge-returns --lower-cf-to-handshake --remove-operation-names %s --split-input-file | FileCheck %s

// a loop-carried value enters the loop header through a mux,
// whose second input is the loop's own cond_br on the comparison
// CHECK-LABEL: handshake.func @selfLoop(
// CHECK-SAME:    %[[IN0:[a-zA-Z0-9_]+]]: !handshake.channel<i32>
// CHECK:         %[[VAL:[a-zA-Z0-9_]+]] = mux %{{[a-zA-Z0-9_]+}} {{\[}}%[[IN0]], %[[BACK:[a-zA-Z0-9_]+]]] {handshake.bb = 1 : ui32}
// CHECK:         %[[CMP:[a-zA-Z0-9_]+]] = cmpi eq, %[[VAL]],
// CHECK:         %[[BACK]], %{{[a-zA-Z0-9_]+}} = cond_br %[[CMP]], %[[VAL]] {handshake.bb = 1 : ui32}
// CHECK:         end {handshake.bb = 2 : ui32}
func.func @selfLoop(%arg0: i32 {handshake.arg_name = "in0"}, %arg1: i32 {handshake.arg_name = "in1"}) {
  cf.br ^bb1(%arg0: i32)
  ^bb1(%0: i32):
    %1 = arith.cmpi eq, %0, %arg1: i32
    cf.cond_br %1, ^bb1(%0: i32), ^bb2
  ^bb2:
    return
}

// -----

// block arguments with no users are not routed
// CHECK-LABEL: handshake.func @duplicateLiveOut(
// CHECK-NOT:     {{mux|merge|cond_br}}
// CHECK:         end {handshake.bb = 1 : ui32}
func.func @duplicateLiveOut(%arg0: i1 {handshake.arg_name = "in0"}, %arg1: i32 {handshake.arg_name = "in1"}, %arg2: i32 {handshake.arg_name = "in2"}) {
  cf.cond_br %arg0, ^bb1(%arg1, %arg2, %arg1: i32, i32, i32), ^bb1(%arg2, %arg2, %arg2: i32, i32, i32)
  ^bb1(%0: i32, %1: i32, %2: i32):
    return
}

// -----

// block arguments with no users are not routed
// CHECK-LABEL: handshake.func @divergeSameArg(
// CHECK-NOT:     {{mux|merge|cond_br}}
// CHECK:         end {handshake.bb = 3 : ui32}
func.func @divergeSameArg(%arg0: i1 {handshake.arg_name = "in0"}, %arg1: i32 {handshake.arg_name = "in1"}) {
  cf.cond_br %arg0, ^bb1(%arg1: i32), ^bb2(%arg1: i32)
  ^bb1(%0: i32):
    cf.br ^bb3
  ^bb2(%1: i32):
    cf.br ^bb3
  ^bb3:
    return
}
