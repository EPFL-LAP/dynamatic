// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL: xls.sproc @end_nop
// CHECK-SAME:    %[[ARG0:.*]]: !xls.schan<i32, in>, %[[ARG1:.*]]: !xls.schan<i32, out>

// CHECK-LABEL:  xls.spawn @nop_32
// CHECK-SAME:     %[[ARG0]], %[[ARG1]]
module {
  handshake.func @end_nop(%in: <i32>) -> (out: <i32>) {
    end %in : <i32>
  }
}
