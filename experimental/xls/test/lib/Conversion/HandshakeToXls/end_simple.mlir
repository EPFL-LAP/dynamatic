// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL: xls.sproc @end_simple
// CHECK-SAME:    %[[ARG0:.*]]: !xls.schan<i32, in>, %[[ARG1:.*]]: !xls.schan<i32, in>, %[[ARG2:.*]]: !xls.schan<i32, out>

// CHECK-LABEL:  xls.spawn @addi_32
// CHECK-SAME:     %[[ARG0]], %[[ARG1]], %[[ARG2]]
module {
  handshake.func @end_simple(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>) -> (!handshake.channel<i32>) attributes {argNames = ["lhs", "rhs"], resNames = ["out"]} {
    %0 = addi %arg0, %arg1 : <i32>
    end %0 : <i32>
  }
}

