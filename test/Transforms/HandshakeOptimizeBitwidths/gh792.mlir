// RUN: dynamatic-opt --handshake-optimize-bitwidths --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @test(
// CHECK-SAME:                         %[[VAL_0:.*]]: !handshake.channel<i2>,
// CHECK-SAME:                         %[[VAL_1:.*]]: !handshake.channel<i32>,
// CHECK-SAME:                         %[[VAL_2:.*]]: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["var0", "var2", "start"], resNames = ["out0", "end"]} {
// CHECK:           %[[VAL_3:.*]] = source : <>
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]] {value = 1 : i2} : <>, <i2>
// CHECK:           %[[VAL_5:.*]] = shrui %[[VAL_0]], %[[VAL_4]] : <i2>
// CHECK:           %[[VAL_6:.*]] = trunci %[[VAL_5]] : <i2> to <i1>
// CHECK:           %[[VAL_7:.*]] = extsi %[[VAL_6]] : <i1> to <i11>
// CHECK:           %[[VAL_8:.*]] = extui %[[VAL_7]] : <i11> to <i32>
// CHECK:           end %[[VAL_8]], %[[VAL_2]] : <i32>, <>
// CHECK:         }
handshake.func @test(%arg0: !handshake.channel<i2>, %arg1: !handshake.channel<i32>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["var0", "var2", "start"], resNames = ["out0", "end"]} {
  %11 = extsi %arg0 : <i2> to <i16>
  %13 = source : <>
  %14 = constant %13 {value = 5 : i16} : <>, <i16>
  %16 = shrui %11, %14 : <i16>
  %17 = extui %16 : <i16> to <i32>
  end %17, %arg2 : <i32>, <>
}
