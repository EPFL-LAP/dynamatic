// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @nop_32(
// CHECK-SAME:                      %[[INP:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i32, in>,
// CHECK-SAME:                      %[[OUT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i32, out>) {

// CHECK:           next (%[[INP]]: !xls.schan<i32, in>, %[[OUT]]: !xls.schan<i32, out>) zeroinitializer {
// CHECK:             %[[TOK0:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TOK1:.*]], %[[DATA:.*]] = xls.sblocking_receive %[[TOK0]], %[[INP]] : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[DATA]], %[[OUT]] : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_nop(%arg0: !handshake.channel<i32>) -> (!handshake.channel<i32>) {
    end %arg0 : <i32>
  }
}
