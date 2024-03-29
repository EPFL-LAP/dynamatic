// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --exp-test-cdg-analysis %s --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test1(
// CHECK-SAME:                     %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i32) {
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1(%[[VAL_1]] : i32), ^bb2(%[[VAL_1]] : i32) {CD = "^bb0 [^bb1 ^bb2 ^bb3 ]"}
// CHECK:         ^bb1(%[[VAL_2:.*]]: i32):
// CHECK:           cf.br ^bb4 {CD = "^bb1 []"}
// CHECK:         ^bb2(%[[VAL_3:.*]]: i32):
// CHECK:           cf.br ^bb3 {CD = "^bb2 []"}
// CHECK:         ^bb3:
// CHECK:           cf.br ^bb4 {CD = "^bb3 []"}
// CHECK:         ^bb4:
// CHECK:           return {CD = "^bb4 []"}
// CHECK:         }
func.func @test1(%arg0: i1, %arg1: i32) {
  cf.cond_br %arg0, ^bb1(%arg1: i32), ^bb2(%arg1: i32)
  ^bb1(%0: i32):
    cf.br ^bb4
  ^bb2(%1: i32):
    cf.br ^bb3
  ^bb3:
    cf.br ^bb4
  ^bb4:
    return
}

// -----

// CHECK-LABEL:   func.func @test2(
// CHECK-SAME:                     %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) {
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2 {CD = "^bb0 [^bb1 ^bb2 ^bb5 ]"}
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb6 {CD = "^bb1 []"}
// CHECK:         ^bb2:
// CHECK:           cf.cond_br %[[VAL_1]], ^bb3, ^bb4 {CD = "^bb2 [^bb3 ^bb4 ]"}
// CHECK:         ^bb3:
// CHECK:           cf.cond_br %[[VAL_1]], ^bb5, ^bb4 {CD = "^bb3 [^bb4 ]"}
// CHECK:         ^bb4:
// CHECK:           cf.br ^bb5 {CD = "^bb4 []"}
// CHECK:         ^bb5:
// CHECK:           cf.br ^bb6 {CD = "^bb5 []"}
// CHECK:         ^bb6:
// CHECK:           return {CD = "^bb6 []"}
// CHECK:         }
func.func @test2(%c0: i1, %c2: i1, %c3: i1) {
  cf.cond_br %c0, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb6
  ^bb2:
    cf.cond_br %c2, ^bb3, ^bb4
  ^bb3:
    cf.cond_br %c2, ^bb5, ^bb4
  ^bb4:
    cf.br ^bb5
  ^bb5:
    cf.br ^bb6
  ^bb6:
    return
}

// -----

// CHECK-LABEL:   func.func @test3(
// CHECK-SAME:                     %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) {
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2 {CD = "^bb0 [^bb1 ^bb5 ^bb2 ]"}
// CHECK:         ^bb1:
// CHECK:           cf.cond_br %[[VAL_1]], ^bb3, ^bb4 {CD = "^bb1 [^bb3 ^bb4 ]"}
// CHECK:         ^bb2:
// CHECK:           cf.cond_br %[[VAL_2]], ^bb4, ^bb6 {CD = "^bb2 [^bb4 ^bb5 ]"}
// CHECK:         ^bb3:
// CHECK:           cf.br ^bb5 {CD = "^bb3 []"}
// CHECK:         ^bb4:
// CHECK:           cf.br ^bb5 {CD = "^bb4 []"}
// CHECK:         ^bb5:
// CHECK:           cf.br ^bb6 {CD = "^bb5 []"}
// CHECK:         ^bb6:
// CHECK:           return {CD = "^bb6 []"}
// CHECK:         }
func.func @test3(%c0: i1, %c2: i1, %c3: i1) {
  cf.cond_br %c0, ^bb1, ^bb2
  ^bb1:
    cf.cond_br %c2, ^bb3, ^bb4
  ^bb2:
    cf.cond_br %c3, ^bb4, ^bb6
  ^bb3:
    cf.br ^bb5
  ^bb4:
    cf.br ^bb5
  ^bb5:
    cf.br ^bb6
  ^bb6:
    return
}
