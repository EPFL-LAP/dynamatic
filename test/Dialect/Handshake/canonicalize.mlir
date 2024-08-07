// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --canonicalize %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @reshapeFoldData(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !handshake.channel<f32, [down1: i2, down2: i8]>, ...) -> !handshake.channel<f32, [down1: i2, down2: i8]> attributes {argNames = ["channel"], resNames = ["out0"]} {
// CHECK:           end %[[VAL_0]] : <f32, [down1: i2, down2: i8]>
// CHECK:         }
handshake.func @reshapeFoldData(%channel: !handshake.channel<f32, [down1: i2, down2: i8]>) -> !handshake.channel<f32, [down1: i2, down2: i8]> {
  %reshaped = reshape [MergeData] %channel : <f32, [down1: i2, down2: i8]> -> <i42>
  %backToOriginal = reshape [SplitData] %reshaped : <i42> -> <f32, [down1: i2, down2: i8]>
  end %backToOriginal : <f32, [down1: i2, down2: i8]>
}

// -----

// CHECK-LABEL:   handshake.func @reshapeFoldExtra(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !handshake.channel<f32, [down1: i2, down2: i8]>, ...) -> !handshake.channel<f32, [down1: i2, down2: i8]> attributes {argNames = ["channel"], resNames = ["out0"]} {
// CHECK:           end %[[VAL_0]] : <f32, [down1: i2, down2: i8]>
// CHECK:         }
handshake.func @reshapeFoldExtra(%channel: !handshake.channel<f32, [down1: i2, down2: i8]>) -> !handshake.channel<f32, [down1: i2, down2: i8]> {
  %reshaped = reshape [MergeExtra] %channel : <f32, [down1: i2, down2: i8]> -> <f32, [mergedDown: i10]>
  %backToOriginal = reshape [SplitExtra] %reshaped : <f32, [mergedDown: i10]> -> <f32, [down1: i2, down2: i8]>
  end %backToOriginal : <f32, [down1: i2, down2: i8]>
}
