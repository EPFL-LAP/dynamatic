// The same output is checked twice:
// SELECT checks that the select input of each mux is the select output
// of a different control merge,
// COUNT checks how many muxes there are
// RUN: dynamatic-opt %s --lower-cf-to-handshake --remove-operation-names --split-input-file | FileCheck %s --check-prefix=SELECT
// RUN: dynamatic-opt %s --lower-cf-to-handshake --remove-operation-names --split-input-file | FileCheck %s --check-prefix=COUNT

// A loop with two values carried around it, and no others
// SELECT-LABEL: handshake.func @loop
// COUNT-LABEL: handshake.func @loop
func.func @loop() {
  %c0 = arith.constant 0 : i32
  cf.br ^bb1(%c0, %c0 : i32, i32)
^bb1(%i: i32, %j: i32):
  // %i and %j, the block arguments of bb1, each become a mux in bb1,
  // and the select input of each mux is the select output
  // of a different control merge in bb1
  // (DAG matches these in any order,
  // and never matches the same line twice)
  // SELECT-DAG: mux %[[SELECT_0:[^ ]+]]
  // SELECT-DAG: %[[SELECT_0]] = control_merge
  // SELECT-DAG: mux %[[SELECT_1:[^ ]+]]
  // SELECT-DAG: %[[SELECT_1]] = control_merge
  // there are exactly two muxes
  // COUNT: mux
  // COUNT: mux
  // COUNT-NOT: mux
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %nextI = arith.addi %i, %c1 : i32
  %nextJ = arith.addi %j, %nextI : i32
  %cond = arith.cmpi slt, %nextI, %c10 : i32
  cf.cond_br %cond, ^bb1(%nextI, %nextJ : i32, i32), ^bb2
^bb2:
  return
}

// -----

// An if-else where each branch produces the two values used after it,
// and no others
// SELECT-LABEL: handshake.func @if_else
// COUNT-LABEL: handshake.func @if_else
func.func @if_else(%arg0: i1 {handshake.arg_name = "in0"}) -> (i32, i32) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  cf.br ^bb3(%c1, %c2 : i32, i32)
^bb2:
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  cf.br ^bb3(%c3, %c4 : i32, i32)
^bb3(%x: i32, %y: i32):
  // %x and %y, the block arguments of bb3, each become a mux in bb3,
  // and the select input of each mux is the select output
  // of a different control merge in bb3
  // (DAG matches these in any order,
  // and never matches the same line twice)
  // SELECT-DAG: mux %[[SELECT_0:[^ ]+]]
  // SELECT-DAG: %[[SELECT_0]] = control_merge
  // SELECT-DAG: mux %[[SELECT_1:[^ ]+]]
  // SELECT-DAG: %[[SELECT_1]] = control_merge
  // there are exactly two muxes
  // COUNT: mux
  // COUNT: mux
  // COUNT-NOT: mux
  return %x, %y : i32, i32
}
