// RUN: dynamatic-opt --handshake-optimize-bitwidths --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @test(
// CHECK-SAME:     %[[VAL_0:.*]]: <i2>, %[[VAL_1:.*]]: <i32>, %[[VAL_2:.*]]: <>) -> (out0: <i32>, end: <>) {
// CHECK:           %[[VAL_3:.*]] = source : <>
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]] {value = 1 : i2} : <>, <i2>
// CHECK:           %[[VAL_5:.*]] = shrui %[[VAL_0]], %[[VAL_4]] : <i2>
// CHECK:           %[[VAL_6:.*]] = trunci %[[VAL_5]] : <i2> to <i1>
// CHECK:           %[[VAL_7:.*]] = extsi %[[VAL_6]] : <i1> to <i11>
// CHECK:           %[[VAL_8:.*]] = extui %[[VAL_7]] : <i11> to <i32>
// CHECK:           end %[[VAL_8]], %[[VAL_2]] : <i32>, <>
// CHECK:         }
handshake.func @test(%var0: <i2>, %var2: <i32>, %start: <>) -> (out0: <i32>, end: <>) {
  %11 = extsi %var0 : <i2> to <i16>
  %13 = source : <>
  %14 = constant %13 {value = 5 : i16} : <>, <i16>
  %16 = shrui %11, %14 : <i16>
  %17 = extui %16 : <i16> to <i32>
  end %17, %start : <i32>, <>
}
