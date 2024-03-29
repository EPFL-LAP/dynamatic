// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --lower-cf-to-handshake --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @selfLoop(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32,
// CHECK-SAME:                             %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "in1"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_1]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_4:.*]] = br %[[VAL_2]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_5:.*]] = br %[[VAL_3]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_6:.*]] = mux %[[VAL_7:.*]] {{\[}}%[[VAL_8:.*]], %[[VAL_4]]] {bb = 1 : ui32} : index, i32
// CHECK:           %[[VAL_9:.*]], %[[VAL_7]] = control_merge %[[VAL_10:.*]], %[[VAL_5]] {bb = 1 : ui32} : none, index
// CHECK:           %[[VAL_11:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_6]] {bb = 1 : ui32} : i32
// CHECK:           %[[VAL_8]], %[[VAL_12:.*]] = cond_br %[[VAL_11]], %[[VAL_6]] {bb = 1 : ui32} : i32
// CHECK:           %[[VAL_10]], %[[VAL_13:.*]] = cond_br %[[VAL_11]], %[[VAL_9]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = control_merge %[[VAL_13]] {bb = 2 : ui32} : none, index
// CHECK:           %[[VAL_16:.*]] = return {bb = 2 : ui32} %[[VAL_14]] : none
// CHECK:           end {bb = 2 : ui32} %[[VAL_16]] : none
// CHECK:         }
func.func @selfLoop(%arg0: i32) {
  cf.br ^bb1(%arg0: i32)
  ^bb1(%0: i32):
    %1 = arith.cmpi eq, %0, %0: i32
    cf.cond_br %1, ^bb1(%0: i32), ^bb2
  ^bb2:
    return
}

// -----

// CHECK-LABEL:   handshake.func @duplicateLiveOut(
// CHECK-SAME:                                     %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                     %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32,
// CHECK-SAME:                                     %[[VAL_3:.*]]: none, ...) -> none attributes {argNames = ["in0", "in1", "in2", "in3"], resNames = ["out0"]} {
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_0]] {bb = 0 : ui32} : i1
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_2]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_7:.*]] = merge %[[VAL_3]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_4]], %[[VAL_5]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_4]], %[[VAL_6]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_4]], %[[VAL_7]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_14:.*]] = mux %[[VAL_15:.*]] {{\[}}%[[VAL_8]], %[[VAL_10]]] {bb = 1 : ui32} : index, i32
// CHECK:           %[[VAL_16:.*]] = mux %[[VAL_15]] {{\[}}%[[VAL_10]], %[[VAL_10]]] {bb = 1 : ui32} : index, i32
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_15]] {{\[}}%[[VAL_8]], %[[VAL_10]]] {bb = 1 : ui32} : index, i32
// CHECK:           %[[VAL_18:.*]], %[[VAL_15]] = control_merge %[[VAL_12]], %[[VAL_12]] {bb = 1 : ui32} : none, index
// CHECK:           %[[VAL_19:.*]] = return {bb = 1 : ui32} %[[VAL_18]] : none
// CHECK:           end {bb = 1 : ui32} %[[VAL_19]] : none
// CHECK:         }
func.func @duplicateLiveOut(%arg0: i1, %arg1: i32, %arg2: i32) {
  cf.cond_br %arg0, ^bb1(%arg1, %arg2, %arg1: i32, i32, i32), ^bb1(%arg2, %arg2, %arg2: i32, i32, i32)
  ^bb1(%0: i32, %1: i32, %2: i32):
    return
}

// ----

// CHECK-LABEL:   handshake.func @divergeSameArg(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                   %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                   %[[VAL_2:.*]]: none, ...) -> none attributes {argNames = ["in0", "in1", "in2"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]] {bb = 0 : ui32} : i1
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_1]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_2]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_3]], %[[VAL_4]] {bb = 0 : ui32} : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_3]], %[[VAL_5]] {bb = 0 : ui32} : none
// CHECK:           %[[VAL_10:.*]] = merge %[[VAL_6]] {bb = 1 : ui32} : i32
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = control_merge %[[VAL_8]] {bb = 1 : ui32} : none, index
// CHECK:           %[[VAL_13:.*]] = br %[[VAL_11]] {bb = 1 : ui32} : none
// CHECK:           %[[VAL_14:.*]] = merge %[[VAL_7]] {bb = 2 : ui32} : i32
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = control_merge %[[VAL_9]] {bb = 2 : ui32} : none, index
// CHECK:           %[[VAL_17:.*]] = br %[[VAL_15]] {bb = 2 : ui32} : none
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = control_merge %[[VAL_17]], %[[VAL_13]] {bb = 3 : ui32} : none, index
// CHECK:           %[[VAL_20:.*]] = return {bb = 3 : ui32} %[[VAL_18]] : none
// CHECK:           end {bb = 3 : ui32} %[[VAL_20]] : none
// CHECK:         }
func.func @divergeSameArg(%arg0: i1, %arg1: i32) {
  cf.cond_br %arg0, ^bb1(%arg1: i32), ^bb2(%arg1: i32)
  ^bb1(%0: i32):
    cf.br ^bb3
  ^bb2(%1: i32):
    cf.br ^bb3
  ^bb3:
    return
}
