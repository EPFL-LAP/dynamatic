// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @source(
// CHECK-SAME:                      %[[OUT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i0, out>) {

// CHECK:           next (%[[OUT]]: !xls.schan<i0, out>) zeroinitializer {
// CHECK:             %[[TOK0:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[VAL:.*]] = "xls.constant_scalar"() <{value = 0 : i32}> : () -> i0
// CHECK:             %{{.*}} = xls.ssend %[[TOK0]], %[[VAL]], %[[OUT]] : (!xls.token, i0, !xls.schan<i0, out>) -> !xls.token
// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_source() -> (!handshake.control<>) {
    %1 = source : <>
    end %1: <>
  }
}
