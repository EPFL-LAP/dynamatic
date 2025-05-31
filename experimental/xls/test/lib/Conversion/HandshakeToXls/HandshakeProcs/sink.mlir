// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @sink_101(
// CHECK-SAME:                        %[[IN:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i101, in>) {

// CHECK:           next (%[[IN]]: !xls.schan<i101, in>) zeroinitializer {
// CHECK:             %[[TOK0:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TOK1:.*]], %[[DATA:.*]] = xls.sblocking_receive %[[TOK0]], %[[IN]] : (!xls.token, !xls.schan<i101, in>) -> (!xls.token, i101)
// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_sink(%in: !handshake.channel<i101>) -> () {
    sink %in : <i101>
    end 
  }
}
