// RUN: dynamatic-opt %s --split-input-file | dynamatic-opt --split-input-file | FileCheck %s

// The arguments are named by their SSA names, and the results inline
// CHECK-LABEL: handshake.func @names(%a: <i32>, %m: memref<4xi32>, %start: <>) -> (out0: <i32>, end: <>) {
// CHECK-NEXT:    end %a, %start : <i32>, <>
handshake.func @names(%a: <i32>, %m: memref<4xi32>, %start: <>) -> (out0: <i32>, end: <>) {
  end %a, %start : <i32>, <>
}

// -----

// Explicit names take precedence over the SSA names and the result names
// CHECK-LABEL: handshake.func @explicitNames(%a: <i32>, %start: <>) -> (out0: <i32>, end: <>) {
// CHECK-NEXT:    end %a, %start : <i32>, <>
handshake.func @explicitNames(%a: <i32>, %start: <>) -> (out0: <i32>, end: <>) {
  end %a, %start : <i32>, <>
}

// -----

// Names that are not unique cannot all be SSA names, so they are printed
// CHECK-LABEL: handshake.func @duplicateNames(%a: <i32>, %a_0: <>) -> (end: <>) attributes {argNames = ["a", "a"]} {
handshake.func @duplicateNames(%x: <i32>, %y: <>) -> (end: <>) attributes {argNames = ["a", "a"]} {
  end %y : <>
}

// -----

// A function without a body has no SSA names
// CHECK-LABEL: handshake.func private @external(<i32>, <>) -> (out0: <i32>, end: <>) attributes {argNames = ["in0", "in1"]}
handshake.func private @external(<i32>, <>) -> (<i32>, end: <>)

// -----

// A single result may be written without parentheses or name
// CHECK-LABEL: handshake.func @singleResult(%start: <>) -> (out0: <>) {
handshake.func @singleResult(%start: <>) -> (<>) {
  end %start : <>
}
